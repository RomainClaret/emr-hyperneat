"""
Base algorithm interface for the unified experiment framework.

This module defines the abstract base class that all algorithm implementations
must inherit from to ensure compatibility with the framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Iterator, Union
from dataclasses import dataclass
import time
import numpy as np
from ..utils.metrics_utils import ensure_dict
from ..adapters.metrics_storage_adapter import (
    storage_registry, get_optimal_adapter
)
from .experiment_reporter import get_global_reporter


@dataclass
class NetworkInfo:
    """Standardized network information across all algorithms."""
    num_nodes: int
    num_connections: int
    num_inputs: int
    num_outputs: int
    num_hidden: int
    connections: List[Tuple[int, int, float]]  # (from, to, weight)
    nodes: Dict[int, Dict[str, Any]]  # node_id -> node properties
    layer_info: Optional[Dict[str, Any]] = None  # For substrate-based algorithms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'num_nodes': self.num_nodes,
            'num_connections': self.num_connections,
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'num_hidden': self.num_hidden,
            'connections': self.connections,
            'nodes': self.nodes,
            'layer_info': self.layer_info
        }


@dataclass
class AlgorithmMetrics:
    """Standardized metrics collected during algorithm execution."""
    generation: int
    best_fitness: float
    mean_fitness: float
    min_fitness: float
    max_fitness: float
    std_fitness: float
    num_species: int
    species_sizes: List[int]
    species_fitness: List[float]
    evaluations: int
    time_elapsed: float
    
    # Normalized fitness in [0, 1] range
    normalized_best_fitness: Optional[float] = None
    
    # Normalized population statistics in [0, 1] range
    normalized_mean_fitness: Optional[float] = None
    normalized_std_fitness: Optional[float] = None
    
    # Population-wide network topology statistics
    hidden_nodes_mean: Optional[float] = None
    hidden_nodes_std: Optional[float] = None
    connections_mean: Optional[float] = None
    connections_std: Optional[float] = None
    density_mean: Optional[float] = None
    density_std: Optional[float] = None
    
    # Diversity metrics
    species_diversity: Optional[float] = None  # Shannon entropy of species distribution
    
    # Mutation tracking metrics
    mutation_counts: Optional[Dict[str, int]] = None  # Count of each mutation type attempted
    mutation_success_rates: Optional[Dict[str, float]] = None  # Success rate per mutation type
    mutation_impact: Optional[Dict[str, float]] = None  # Average fitness impact per mutation type
    total_mutations_attempted: int = 0  # Total mutations attempted this generation
    total_mutations_successful: int = 0  # Total successful mutations this generation
    mutation_efficiency: float = 0.0  # successful_mutations / total_attempted
    
    # Algorithm-specific metrics
    custom_metrics: Dict[str, Any] = None

    # Performance profiling statistics (when profiled implementation is used)
    profile_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Helper function to convert LazyMetricValue objects to regular Python values
        def convert_value(value):
            """Convert LazyMetricValue or other types to serializable form."""
            if value is None:
                return None
            # Check if it has __float__ method (LazyMetricValue from TensorNEAT)
            if hasattr(value, '__float__') and not isinstance(value, (int, float)):
                return float(value)
            # Handle lists that might contain LazyMetricValue objects
            if isinstance(value, list):
                return [convert_value(v) for v in value]
            # Handle dictionaries recursively
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            return value
        
        result = {
            'generation': convert_value(self.generation),
            'best_fitness': convert_value(self.best_fitness),
            'mean_fitness': convert_value(self.mean_fitness),
            'min_fitness': convert_value(self.min_fitness),
            'max_fitness': convert_value(self.max_fitness),
            'std_fitness': convert_value(self.std_fitness),
            'num_species': convert_value(self.num_species),
            'species_sizes': convert_value(self.species_sizes),
            'species_fitness': convert_value(self.species_fitness),
            'evaluations': convert_value(self.evaluations),
            'time_elapsed': convert_value(self.time_elapsed)
        }
        
        # Add shorter aliases for HTML components compatibility
        # These are the names expected by visualization components
        result['best'] = result['best_fitness']
        result['mean'] = result['mean_fitness']
        result['avg'] = result['mean_fitness']  # Some components use 'avg' instead of 'mean'
        result['min'] = result['min_fitness']
        result['max'] = result['max_fitness']
        result['std'] = result['std_fitness']
        
        # Add normalized fitness if available
        if self.normalized_best_fitness is not None:
            converted = convert_value(self.normalized_best_fitness)
            result['normalized_best_fitness'] = converted
            result['normalized_best'] = converted  # Alias for HTML
        
        # Add normalized population statistics if available
        if self.normalized_mean_fitness is not None:
            converted = convert_value(self.normalized_mean_fitness)
            result['normalized_mean_fitness'] = converted
            result['normalized_mean'] = converted  # Alias for HTML
            result['normalized_avg'] = converted  # Alternative alias
        if self.normalized_std_fitness is not None:
            converted = convert_value(self.normalized_std_fitness)
            result['normalized_std_fitness'] = converted
            result['normalized_std'] = converted  # Alias for HTML
        
        # Add population-wide statistics if available
        if self.hidden_nodes_mean is not None:
            result['hidden_nodes_mean'] = convert_value(self.hidden_nodes_mean)
        if self.hidden_nodes_std is not None:
            result['hidden_nodes_std'] = convert_value(self.hidden_nodes_std)
        if self.connections_mean is not None:
            result['connections_mean'] = convert_value(self.connections_mean)
        if self.connections_std is not None:
            result['connections_std'] = convert_value(self.connections_std)
        if self.density_mean is not None:
            result['density_mean'] = convert_value(self.density_mean)
        if self.density_std is not None:
            result['density_std'] = convert_value(self.density_std)
        if self.species_diversity is not None:
            result['species_diversity'] = convert_value(self.species_diversity)
        
        # Add mutation tracking metrics if available
        if self.mutation_counts is not None:
            result['mutation_counts'] = convert_value(self.mutation_counts)
        if self.mutation_success_rates is not None:
            result['mutation_success_rates'] = convert_value(self.mutation_success_rates)
        if self.mutation_impact is not None:
            result['mutation_impact'] = convert_value(self.mutation_impact)
        
        # Always include these mutation metrics (they have defaults)
        result['total_mutations_attempted'] = convert_value(self.total_mutations_attempted)
        result['total_mutations_successful'] = convert_value(self.total_mutations_successful)
        result['mutation_efficiency'] = convert_value(self.mutation_efficiency)

        if self.custom_metrics:
            result['custom_metrics'] = convert_value(self.custom_metrics)

        # Add performance profiling statistics if available
        if self.profile_stats is not None:
            result['profile_stats'] = convert_value(self.profile_stats)

        return result


class BaseAlgorithm(ABC):
    """Abstract base class for all neuroevolution algorithms."""
    
    def __init__(self, name: str, implementation: str, storage_adapter: Optional[str] = None,
                 rebuild_jax: bool = False, trial_id: int = None):
        """
        Initialize base algorithm.

        Args:
            name: Algorithm name (e.g., 'neat', 'hyperneat', 'eshyperneat')
            implementation: Implementation name (e.g., 'pureples', 'tensorneat')
            storage_adapter: Name of storage adapter to use ('memory', 'cached', 'streaming', or None for auto)
            rebuild_jax: Force JAX cache rebuild before initialization (default: False)
            trial_id: Trial identifier for grid search isolation - creates unique JAX cache per trial (default: None)
        """
        # Setup JAX cache isolation FIRST (before any other initialization)
        # This ensures isolated cache directories for parallel grid search trials
        if trial_id is not None or rebuild_jax:
            self._setup_jax_cache_isolation(implementation, trial_id, rebuild_jax)

        self.name = name
        self.implementation = implementation
        self.trial_id = trial_id
        self.current_generation = 0
        self.callbacks = []
        self.generation_history_manager = None

        # Initialize metrics storage adapter
        self._storage_adapter_name = storage_adapter
        self._metrics_storage = None  # Will be initialized in run_experiment

    def _setup_jax_cache_isolation(self, implementation: str, trial_id: Optional[int], rebuild_jax: bool):
        """Setup JAX cache isolation for grid search trials.

        Creates unique cache directory per implementation, process, and trial to prevent
        JAX memory leaks and cache conflicts during parallel grid search execution.

        Args:
            implementation: Implementation name for cache directory naming
            trial_id: Trial identifier (None if not in grid search)
            rebuild_jax: Whether to clear existing cache

        Note:
            Handles missing JAX gracefully - becomes no-op for non-JAX implementations.
            This allows PUREPLES-only implementations to ignore JAX-specific setup.
        """
        try:
            import os
            import tempfile
            import shutil
            import jax

            # Create unique cache directory per trial
            cache_suffix = f"{implementation}_{os.getpid()}"
            if trial_id is not None:
                cache_suffix += f"_trial{trial_id}"

            cache_dir = os.path.join(
                tempfile.gettempdir(),
                f"jax_cache_{cache_suffix}"
            )

            # Clear existing cache if rebuild requested
            if rebuild_jax and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

            os.makedirs(cache_dir, exist_ok=True)

            # Configure JAX to use isolated cache
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

            # Clear in-memory caches if rebuilding
            if rebuild_jax:
                jax.clear_caches()
                if hasattr(jax, 'clear_backends'):
                    try:
                        jax.clear_backends()
                    except:
                        pass  # Not available in all JAX versions

                import gc
                gc.collect()

        except ImportError:
            # JAX not available - no-op (fine for PUREPLES-only implementations)
            pass

    @abstractmethod
    def create_config(self, params: Dict[str, Any]) -> Any:
        """
        Create algorithm-specific configuration object.
        
        Args:
            params: Dictionary of configuration parameters
            
        Returns:
            Algorithm-specific configuration object
        """
        pass
        
    @abstractmethod
    def initialize(self, config: Any, problem: 'BaseProblem', seed: Optional[int] = None) -> Any:
        """
        Initialize algorithm state.
        
        Args:
            config: Algorithm configuration
            problem: Problem instance
            seed: Random seed for reproducibility
            
        Returns:
            Initial algorithm state
        """
        pass
        
    @abstractmethod
    def run_generation(self, state: Any, problem: 'BaseProblem') -> Tuple[Any, AlgorithmMetrics]:
        """
        Run one generation of the algorithm.
        
        Args:
            state: Current algorithm state
            problem: Problem instance
            
        Returns:
            Tuple of (new_state, metrics)
        """
        pass
        
    @abstractmethod
    def evaluate_genome(self, genome: Any, problem: 'BaseProblem') -> float:
        """
        Evaluate a single genome on the problem.
        
        Args:
            genome: Genome to evaluate
            problem: Problem instance
            
        Returns:
            Fitness value
        """
        pass
        
    @abstractmethod
    def get_best_genome(self, state: Any) -> Any:
        """
        Extract best genome from current state.
        
        Args:
            state: Current algorithm state
            
        Returns:
            Best genome
        """
        pass
        
    @abstractmethod
    def extract_network_info(self, genome: Any) -> NetworkInfo:
        """
        Extract network structure information from genome.
        
        Args:
            genome: Genome to analyze
            
        Returns:
            NetworkInfo object with standardized network information
        """
        pass
        
    @abstractmethod
    def genome_to_phenotype(self, genome: Any) -> Any:
        """
        Convert genome to phenotype network.
        
        Args:
            genome: Genome to convert
            
        Returns:
            Phenotype network that can be used for evaluation
        """
        pass

    def get_config_metadata(self) -> Optional[Dict[str, Any]]:
        """Return configuration metadata including loaded files and hierarchical config.

        This method provides access to the complete configuration hierarchy and metadata
        from ConfigManager. It is used by BaseExperiment to save config_hierarchical.json
        for reproducibility and debugging.

        The default implementation returns metadata stored during create_config() if available.
        Implementations should store config metadata in create_config() like this:

            config_metadata = config.pop('_metadata', {})
            self._config_metadata = config_metadata

        Returns:
            Dictionary containing:
            - '_metadata': Metadata from ConfigManager (loaded_files, load_errors, etc.)
            - Hierarchical configuration structure (population, mutation, weights, etc.)

            Returns None if no metadata available (legacy implementations or if not using ConfigManager).
        """
        # Default implementation returns stored metadata if available
        return getattr(self, '_config_metadata', None)

    def add_callback(self, callback):
        """Add a callback to be called during evolution."""
        self.callbacks.append(callback)
    
    def set_generation_history_manager(self, manager):
        """Set the generation history manager for tracking evolution.
        
        Args:
            manager: GenerationHistoryManager instance
        """
        self.generation_history_manager = manager
    
    @property
    def metrics_history(self) -> Union[List, Iterator]:
        """Backward-compatible property for accessing metrics history.
        
        Returns an iterator over metrics if using cached storage,
        or a list if using memory storage.
        """
        if self._metrics_storage is None:
            return []
        return self._metrics_storage
    
    @metrics_history.setter
    def metrics_history(self, value):
        """Setter for backward compatibility.
        
        When algorithms try to set metrics_history = [], 
        we clear the storage adapter instead.
        """
        if isinstance(value, list) and len(value) == 0:
            # If setting to empty list, clear the storage
            if self._metrics_storage is not None:
                self._metrics_storage.clear()
        else:
            # For non-empty values, log a warning
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning("Direct assignment to metrics_history is deprecated. Use storage adapter.")
    
    def run_experiment(self, config: Any, problem: 'BaseProblem', 
                      max_generations: int, target_fitness: Optional[float] = None,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete experiment (can be overridden for algorithm-specific needs).
        
        Args:
            config: Algorithm configuration
            problem: Problem instance
            max_generations: Maximum generations to run
            target_fitness: Optional target fitness to stop early
            seed: Random seed
            
        Returns:
            Dictionary with results
        """
        # Log entry parameters if logger available
        if hasattr(self, 'logger'):
            self.logger.debug(f"BaseAlgorithm.run_experiment called with max_generations={max_generations}, type={type(max_generations)}")
        
        # Store original target for success determination
        original_target_fitness = target_fitness
        
        # Set the implementation on the problem BEFORE checking fitness range
        # This is crucial for getting the correct fitness range for conversion
        if hasattr(problem, 'set_current_implementation'):
            problem.set_current_implementation(self.implementation)
            if hasattr(self, 'logger'):
                self.logger.debug(f"Set problem implementation to {self.implementation}")
        
        # Convert target_fitness if using adapter and it appears to be normalized
        if target_fitness is not None and hasattr(self, 'adapter') and self.adapter is not None:
            # Check if target_fitness appears to be in normalized scale (0-1 range)
            if 0.0 <= target_fitness <= 1.0:
                original_target = target_fitness
                # Get fitness range from problem for proper conversion
                fitness_range = (0.0, 1.0)  # Default
                if hasattr(problem, 'get_fitness_range'):
                    fitness_range = problem.get_fitness_range()
                # Convert to implementation-specific scale
                if hasattr(self.adapter, 'get_fitness_threshold'):
                    target_fitness = self.adapter.get_fitness_threshold(target_fitness, fitness_range)
                elif hasattr(self.adapter, 'denormalize_fitness'):
                    target_fitness = self.adapter.denormalize_fitness(target_fitness, fitness_range)
                
                if hasattr(self, 'logger'):
                    self.logger.debug(
                        f"BaseAlgorithm converted target_fitness from {original_target:.6f} to {target_fitness:.6f} "
                        f"for {self.name}/{self.implementation} (range: {fitness_range})"
                    )
        
        # Initialize
        state = self.initialize(config, problem, seed)
        self.current_generation = 0
        
        # Initialize metrics storage adapter
        if self._metrics_storage is None:
            # Auto-select adapter if not specified
            if self._storage_adapter_name is None:
                self._storage_adapter_name = get_optimal_adapter(
                    max_generations=max_generations,
                    population_size=config.get('population_size', 150) if hasattr(config, 'get') else 150
                )
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Auto-selected storage adapter: {self._storage_adapter_name} (was None)")
            else:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Using pre-set storage adapter: {self._storage_adapter_name}")
            
            # Create storage adapter
            storage_kwargs = {}
            if self._storage_adapter_name in ['cached', 'streaming']:
                # Check if we have a variant_name attribute for cache directory naming
                variant_suffix = getattr(self, 'variant_name', self.implementation)
                storage_kwargs = {
                    'memory_size': 10,  # Keep last 10 generations in memory
                    'cache_dir': f'.emr_cache/metrics/{self.name}_{variant_suffix}'
                }
            elif self._storage_adapter_name == 'stream_only':
                # StreamOnlyAdapter uses window_size parameter
                variant_suffix = getattr(self, 'variant_name', self.implementation)
                storage_kwargs = {
                    'window_size': 10,  # Keep last 10 generations in memory
                    'cache_dir': f'.emr_cache/metrics/{self.name}_{variant_suffix}'
                }
            
            self._metrics_storage = storage_registry.create(
                self._storage_adapter_name,
                **storage_kwargs
            )
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Initialized {self._storage_adapter_name} storage adapter")
        
        # Clear any existing metrics
        self._metrics_storage.clear()
        
        # Run evolution
        winner = None
        start_time = time.time()
        
        # Log evolution start if logger available
        if hasattr(self, 'logger'):
            self.logger.info(f"Starting evolution loop: current_generation={self.current_generation}, max_generations={max_generations}, type(max_generations)={type(max_generations)}")
        
        while self.current_generation < max_generations:
            if hasattr(self, 'logger'):
                self.logger.info(f"TOP OF LOOP: Generation {self.current_generation}: checking {self.current_generation} < {max_generations}, will_continue={self.current_generation < max_generations}")

            try:
                state, metrics = self.run_generation(state, problem)
                self._metrics_storage.append(metrics)

                # Log metrics recording
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Recorded metrics for generation {self.current_generation}: best_fitness={metrics.best_fitness if hasattr(metrics, 'best_fitness') else 'N/A'}")
                    self.logger.debug(f"Metrics storage now has {len(self._metrics_storage._history) if hasattr(self._metrics_storage, '_history') else 'unknown'} entries")

                self.current_generation += 1

                # Report progress to experiment manager if available
                # Check if real-time events are enabled (default: False for backward compatibility)
                # These values are set on the algorithm instance by base_experiment.py before create_config()
                enable_realtime_events = getattr(self, 'enable_realtime_events', False)
                batch_transfers = getattr(self, 'batch_device_transfers', False)
                event_interval = getattr(self, 'event_emission_interval', 1)

                # CRITICAL DEBUG: Unconditional logging to verify attribute reading
                import logging as log_module
                base_algo_logger = log_module.getLogger('base_algorithm_debug')
                base_algo_logger.warning(
                    f"[BASE_ALGORITHM DEBUG] Gen {self.current_generation}: "
                    f"Read enable_realtime_events={enable_realtime_events}, "
                    f"batch_device_transfers={batch_transfers}, "
                    f"event_emission_interval={event_interval}, "
                    f"hasattr_enable={hasattr(self, 'enable_realtime_events')}, "
                    f"hasattr_batch={hasattr(self, 'batch_device_transfers')}"
                )

                # Determine if we should emit this generation
                # Always emit: first gen, multiples of interval, and final generation
                is_first_generation = self.current_generation == 1
                is_interval_generation = (self.current_generation % event_interval == 0)
                is_final_generation = (self.current_generation >= max_generations)

                should_emit = is_first_generation or is_interval_generation or is_final_generation

                reporter = get_global_reporter()
                base_algo_logger.warning(
                    f"[BASE_ALGORITHM DEBUG] Gen {self.current_generation}: "
                    f"reporter={reporter is not None}, "
                    f"should_emit={should_emit} (first={is_first_generation}, interval={is_interval_generation}, final={is_final_generation}), "
                    f"Will emit events: {reporter is not None and enable_realtime_events and should_emit}"
                )

                if hasattr(self, 'logger'):
                    self.logger.info(f"[DEBUG REPORTER] gen={self.current_generation}, got reporter: {reporter is not None}, enable_realtime_events={enable_realtime_events}, should_emit={should_emit}")

                if reporter and enable_realtime_events and should_emit:
                    try:
                        # Collect all metrics that need device transfer for batching
                        # This reduces 8 separate GPU→CPU transfers to 1 batched transfer
                        if batch_transfers:
                            # Import jax for batched device_get
                            try:
                                import jax
                                import jax.numpy as jnp

                                # Collect LazyMetricValue objects for batched transfer
                                metrics_to_transfer = {}
                                metric_keys = []

                                # Best fitness
                                if hasattr(metrics.best_fitness, '_jax_value'):
                                    metrics_to_transfer['best_fitness'] = metrics.best_fitness._jax_value
                                    metric_keys.append('best_fitness')

                                # Mean fitness
                                if hasattr(metrics.mean_fitness, '_jax_value'):
                                    metrics_to_transfer['mean_fitness'] = metrics.mean_fitness._jax_value
                                    metric_keys.append('mean_fitness')

                                # Std fitness
                                if hasattr(metrics.std_fitness, '_jax_value'):
                                    metrics_to_transfer['std_fitness'] = metrics.std_fitness._jax_value
                                    metric_keys.append('std_fitness')

                                # Normalized best fitness
                                if hasattr(metrics, 'normalized_best_fitness') and metrics.normalized_best_fitness is not None:
                                    if hasattr(metrics.normalized_best_fitness, '_jax_value'):
                                        metrics_to_transfer['normalized_best_fitness'] = metrics.normalized_best_fitness._jax_value
                                        metric_keys.append('normalized_best_fitness')

                                # Normalized mean fitness
                                if hasattr(metrics, 'normalized_mean_fitness') and metrics.normalized_mean_fitness is not None:
                                    if hasattr(metrics.normalized_mean_fitness, '_jax_value'):
                                        metrics_to_transfer['normalized_mean_fitness'] = metrics.normalized_mean_fitness._jax_value
                                        metric_keys.append('normalized_mean_fitness')

                                # Species count
                                if hasattr(metrics, 'num_species') and metrics.num_species is not None:
                                    if hasattr(metrics.num_species, '_jax_value'):
                                        metrics_to_transfer['num_species'] = metrics.num_species._jax_value
                                        metric_keys.append('num_species')

                                # Network complexity (nodes/connections)
                                if hasattr(metrics, 'custom_metrics') and metrics.custom_metrics:
                                    if 'network_complexity' in metrics.custom_metrics:
                                        complexity = metrics.custom_metrics['network_complexity']
                                        if 'nodes' in complexity and hasattr(complexity['nodes'], '_jax_value'):
                                            metrics_to_transfer['nodes'] = complexity['nodes']._jax_value
                                            metric_keys.append('nodes')
                                        if 'connections' in complexity and hasattr(complexity['connections'], '_jax_value'):
                                            metrics_to_transfer['connections'] = complexity['connections']._jax_value
                                            metric_keys.append('connections')

                                # Single batched device transfer (8 transfers → 1 transfer!)
                                if metrics_to_transfer:
                                    transferred = jax.device_get(metrics_to_transfer)

                                    # Convert transferred values to pure Python types
                                    best_fitness = float(transferred['best_fitness']) if 'best_fitness' in transferred else float(metrics.best_fitness)
                                    mean_fitness = float(transferred['mean_fitness']) if 'mean_fitness' in transferred else float(metrics.mean_fitness)
                                    std_fitness = float(transferred['std_fitness']) if 'std_fitness' in transferred else (float(metrics.std_fitness) if metrics.std_fitness is not None else None)
                                    normalized_best_fitness = float(transferred['normalized_best_fitness']) if 'normalized_best_fitness' in transferred else None
                                    normalized_mean_fitness = float(transferred['normalized_mean_fitness']) if 'normalized_mean_fitness' in transferred else None
                                    species_count = int(float(transferred['num_species'])) if 'num_species' in transferred else None
                                    nodes = int(float(transferred['nodes'])) if 'nodes' in transferred else None
                                    connections = int(float(transferred['connections'])) if 'connections' in transferred else None
                                else:
                                    # No LazyMetricValue objects, use values directly
                                    best_fitness = float(metrics.best_fitness)
                                    mean_fitness = float(metrics.mean_fitness)
                                    std_fitness = float(metrics.std_fitness) if metrics.std_fitness is not None else None
                                    normalized_best_fitness = float(metrics.normalized_best_fitness) if hasattr(metrics, 'normalized_best_fitness') and metrics.normalized_best_fitness is not None else None
                                    normalized_mean_fitness = float(metrics.normalized_mean_fitness) if hasattr(metrics, 'normalized_mean_fitness') and metrics.normalized_mean_fitness is not None else None
                                    species_count = int(metrics.num_species) if hasattr(metrics, 'num_species') and metrics.num_species is not None else None
                                    nodes = None
                                    connections = None
                                    if hasattr(metrics, 'custom_metrics') and metrics.custom_metrics and 'network_complexity' in metrics.custom_metrics:
                                        complexity = metrics.custom_metrics['network_complexity']
                                        nodes = int(float(complexity['nodes'])) if 'nodes' in complexity else None
                                        connections = int(float(complexity['connections'])) if 'connections' in complexity else None

                            except ImportError:
                                # JAX not available, fall back to individual conversions
                                batch_transfers = False

                        if not batch_transfers:
                            # Fall back to individual conversions (original behavior)
                            best_fitness = metrics.best_fitness
                            if hasattr(best_fitness, '__float__'):
                                best_fitness = float(best_fitness)

                            mean_fitness = metrics.mean_fitness
                            if hasattr(mean_fitness, '__float__'):
                                mean_fitness = float(mean_fitness)

                            std_fitness = metrics.std_fitness
                            if hasattr(std_fitness, '__float__'):
                                std_fitness = float(std_fitness)
                            else:
                                std_fitness = None

                            # Extract NORMALIZED fitness separately
                            normalized_best_fitness = None
                            normalized_mean_fitness = None
                            if hasattr(metrics, 'normalized_best_fitness') and metrics.normalized_best_fitness is not None:
                                normalized_best_fitness = metrics.normalized_best_fitness
                                if hasattr(normalized_best_fitness, '__float__'):
                                    normalized_best_fitness = float(normalized_best_fitness)

                            if hasattr(metrics, 'normalized_mean_fitness') and metrics.normalized_mean_fitness is not None:
                                normalized_mean_fitness = metrics.normalized_mean_fitness
                                if hasattr(normalized_mean_fitness, '__float__'):
                                    normalized_mean_fitness = float(normalized_mean_fitness)

                            # Extract network complexity if available
                            nodes = None
                            connections = None
                            if hasattr(metrics, 'custom_metrics') and metrics.custom_metrics:
                                if 'network_complexity' in metrics.custom_metrics:
                                    complexity = metrics.custom_metrics['network_complexity']
                                    nodes = complexity.get('nodes')
                                    connections = complexity.get('connections')
                                    # Convert LazyMetricValue if needed
                                    if hasattr(nodes, '__float__'):
                                        nodes = int(float(nodes))
                                    if hasattr(connections, '__float__'):
                                        connections = int(float(connections))

                            # Extract species_count and convert LazyMetricValue to int
                            species_count = None
                            if hasattr(metrics, 'num_species') and metrics.num_species is not None:
                                if hasattr(metrics.num_species, '__float__'):
                                    # LazyMetricValue - convert to int via float
                                    species_count = int(float(metrics.num_species))
                                elif hasattr(metrics.num_species, '__int__'):
                                    # Has __int__ method
                                    species_count = int(metrics.num_species)
                                else:
                                    # Already an int or convertible
                                    species_count = metrics.num_species

                        if hasattr(self, 'logger'):
                            self.logger.info(f"[DEBUG REPORTER] About to call report_progress: best={best_fitness}, mean={mean_fitness}, nodes={nodes}, conns={connections}, species={species_count}")

                        # Report the progress with BOTH raw and normalized fitness
                        reporter.report_progress(
                            generation=self.current_generation,
                            total_generations=max_generations,
                            best_fitness=best_fitness,
                            mean_fitness=mean_fitness,
                            std_fitness=std_fitness,
                            nodes=nodes,
                            connections=connections,
                            species_count=species_count,
                            normalized_best_fitness=normalized_best_fitness,
                            normalized_mean_fitness=normalized_mean_fitness
                        )

                        if hasattr(self, 'logger'):
                            self.logger.info(f"[DEBUG REPORTER] report_progress() completed successfully")
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"[DEBUG REPORTER] Exception in report_progress(): {type(e).__name__}: {e}")
                            import traceback
                            self.logger.error(f"[DEBUG REPORTER] Traceback:\n{traceback.format_exc()}")
                elif reporter and not enable_realtime_events:
                    # Real-time events disabled - skip conversion and emission entirely
                    # Metrics stay on GPU for maximum performance
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"Real-time events disabled - skipping events.jsonl emission (performance mode)")

                # Save generation history if manager is available
                if self.generation_history_manager:
                    try:
                        best_genome = self.get_best_genome(state)
                        metrics_dict = metrics.to_dict()
                        metrics_dict['num_inputs'] = problem.input_size
                        metrics_dict['num_outputs'] = problem.output_size
                        
                        # Add normalized fitness if adapter is available
                        # NOTE: For TensorNEAT implementations, skip normalization here
                        # Let grid_search_engine.py handle it (it has the correct logic)
                        if hasattr(self, 'adapter') and self.adapter is not None:
                            if hasattr(self.adapter, 'normalize_fitness'):
                                implementation = getattr(self, 'implementation', '')
                                is_tensorneat = any(x in implementation.lower() for x in ['tensorneat', 'jax-steps', 'manual-steps', 'compiled'])

                                if not is_tensorneat:
                                    # Only normalize for non-TensorNEAT implementations (PUREPLES, etc.)
                                    fitness_range = (0.0, 1.0)  # Default
                                    if hasattr(problem, 'get_fitness_range'):
                                        fitness_range = problem.get_fitness_range()

                                    normalized_fitness = self.adapter.normalize_fitness(metrics.best_fitness, fitness_range)
                                    metrics_dict['normalized_best_fitness'] = normalized_fitness
                                    metrics_dict['normalized_fitness'] = normalized_fitness
                                    if hasattr(self, 'logger'):
                                        self.logger.debug(
                                            f"Added normalized_best_fitness: {normalized_fitness:.6f} "
                                            f"(from raw: {metrics.best_fitness:.6f}, range: {fitness_range})"
                                        )
                                else:
                                    # TensorNEAT: Let grid_search_engine handle normalization
                                    if hasattr(self, 'logger'):
                                        self.logger.debug(
                                            f"Skipping normalization for TensorNEAT - grid_search_engine will handle it"
                                        )
                        
                        self.generation_history_manager.save_generation(
                            generation=self.current_generation - 1,  # Use the actual generation number
                            population=state,
                            best_genome=best_genome,
                            metrics=metrics_dict,
                            implementation=self.implementation
                        )
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to save generation history: {e}")
                            
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Exception in run_generation: {e}")
                raise
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Completed generation, current_generation now = {self.current_generation}, max_generations still = {max_generations}, continuing? {self.current_generation < max_generations}")
            
            # Fire callbacks for real-time updates
            if self.callbacks:
                # Create partial result for callbacks
                partial_result = {
                    'trial_id': -1,  # Indicates in-progress
                    'success': False,
                    'generations': self.current_generation,
                    'final_fitness': metrics.best_fitness,
                    'total_time': time.time() - start_time,
                    'winner_genome': None,
                    'network_info': {},
                    'metrics_history': self.metrics_history,
                    'metrics': ensure_dict(getattr(metrics, 'custom_metrics', {}))
                }
                
                # Convert to object-like structure for compatibility
                class PartialResult:
                    def __init__(self, data):
                        for k, v in data.items():
                            setattr(self, k, v)
                
                partial_result_obj = PartialResult(partial_result)
                
                # Fire callbacks
                for callback in self.callbacks:
                    try:
                        callback(None, partial_result_obj)
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Callback failed: {e}")
            
            # Check for target fitness
            if hasattr(self, 'logger'):
                self.logger.debug(f"Checking target fitness: target_fitness={target_fitness}")
            if target_fitness is not None:
                if hasattr(self, 'logger'):
                    self.logger.debug(
                        f"Generation {self.current_generation}: best_fitness={metrics.best_fitness:.6f}, "
                        f"target={target_fitness:.6f}, meets_target={metrics.best_fitness >= target_fitness}"
                    )
                if metrics.best_fitness >= target_fitness:
                    winner = self.get_best_genome(state)
                    if hasattr(self, 'logger'):
                        self.logger.info(
                            f"Target fitness reached at generation {self.current_generation}: "
                            f"{metrics.best_fitness:.6f} >= {target_fitness:.6f}"
                        )
                    break
            
            # Log end of loop iteration if logger available
            if hasattr(self, 'logger'):
                self.logger.info(f"END OF LOOP BODY: About to re-check condition: {self.current_generation} < {max_generations}")
        
        # Log loop exit if logger available
        if hasattr(self, 'logger'):
            self.logger.info(f"Exited evolution loop. Final current_generation={self.current_generation}, max_generations={max_generations}")
                
        # Get final best genome if not already found
        if winner is None:
            winner = self.get_best_genome(state)
            
        # Save final generation if not already saved
        if self.generation_history_manager and self.current_generation > 0:
            # Check if the final generation was saved
            final_gen = self.current_generation - 1
            saved_generations = self.generation_history_manager.get_saved_generations()
            
            # Check if final generation is in the list of saved generations
            if final_gen not in saved_generations:
                # Final generation wasn't saved, force save it
                if hasattr(self, 'logger'):
                    self.logger.info(f"Saving final generation {final_gen} to complete animation data")
                
                try:
                    # Get the last metrics
                    final_metrics = self.metrics_history[-1] if self.metrics_history else None
                    if final_metrics:
                        metrics_dict = final_metrics.to_dict()
                        metrics_dict['num_inputs'] = problem.input_size
                        metrics_dict['num_outputs'] = problem.output_size
                        
                        # Add normalized fitness if adapter is available
                        # NOTE: For TensorNEAT implementations, skip normalization here
                        # Let grid_search_engine.py handle it (it has the correct logic)
                        if hasattr(self, 'adapter') and self.adapter is not None:
                            if hasattr(self.adapter, 'normalize_fitness'):
                                implementation = getattr(self, 'implementation', '')
                                is_tensorneat = any(x in implementation.lower() for x in ['tensorneat', 'jax-steps', 'manual-steps', 'compiled'])

                                if not is_tensorneat:
                                    # Only normalize for non-TensorNEAT implementations (PUREPLES, etc.)
                                    fitness_range = (0.0, 1.0)  # Default
                                    if hasattr(problem, 'get_fitness_range'):
                                        fitness_range = problem.get_fitness_range()

                                    normalized_fitness = self.adapter.normalize_fitness(final_metrics.best_fitness, fitness_range)
                                    metrics_dict['normalized_best_fitness'] = normalized_fitness
                                    metrics_dict['normalized_fitness'] = normalized_fitness
                                    if hasattr(self, 'logger'):
                                        self.logger.debug(
                                            f"Final gen: Added normalized_best_fitness: {normalized_fitness:.6f} "
                                            f"(from raw: {final_metrics.best_fitness:.6f}, range: {fitness_range})"
                                        )
                                else:
                                    # TensorNEAT: Let grid_search_engine handle normalization
                                    if hasattr(self, 'logger'):
                                        self.logger.debug(
                                            f"Final gen: Skipping normalization for TensorNEAT - grid_search_engine will handle it"
                                        )

                        # Temporarily override the save check to force saving
                        original_method = self.generation_history_manager.should_save_generation
                        self.generation_history_manager.should_save_generation = lambda g: True
                        
                        try:
                            self.generation_history_manager.save_generation(
                                generation=final_gen,
                                population=state,
                                best_genome=winner,
                                metrics=metrics_dict,
                                implementation=self.implementation
                            )
                        finally:
                            # Restore the original method
                            self.generation_history_manager.should_save_generation = original_method
                            
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Failed to save final generation history: {e}")
            
        # Extract network info properly
        # Note: winner can be a JAX array, dict, or None - check carefully
        network_info = None
        if winner is not None:
            try:
                if isinstance(winner, dict) and 'genome' in winner:
                    # Winner is dict with genome key
                    network_info = self.extract_network_info(winner['genome'])
                else:
                    # Winner is raw genome (could be JAX array or other type)
                    network_info = self.extract_network_info(winner)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Could not extract network info from winner: {e}")

        # CRITICAL: Collect all metrics into a list ONCE to avoid iterator consumption
        # StreamOnlyAdapter returns an iterator that can only be consumed once
        # Creating the list here ensures all subsequent operations use the same data
        metrics_list = list(self.metrics_history) if self.metrics_history else []

        # Extract best fitness - always use historical maximum across all generations
        # Some implementations (PUREPLES) include 'fitness' in winner dict, but that's
        # only the best from the FINAL generation, not the overall best. We must use
        # get_best_fitness() which returns max(m.best_fitness for m in metrics_history)
        best_fitness = 0.0
        if metrics_list:
            # Calculate best fitness directly from metrics_list (no iterator consumption)
            def _to_float(value):
                if hasattr(value, 'value'):
                    if callable(value.value):
                        return float(value.value())
                    else:
                        return float(value.value)
                elif hasattr(value, '__float__'):
                    return float(value)
                else:
                    return float(value)

            best_fitness = max(_to_float(m.best_fitness) for m in metrics_list)

            # Log after getting the value
            if hasattr(self, 'logger'):
                metrics_count = len(metrics_list)
                self.logger.info(
                    f"SUCCESS CHECK [{self.name}/{self.implementation}]: "
                    f"Using metrics_list with {metrics_count} metrics → {best_fitness:.6f}"
                )
        else:
            # Fallback: If no metrics history, try winner['fitness'] as last resort
            if winner and isinstance(winner, dict) and 'fitness' in winner:
                best_fitness = winner['fitness']
                if hasattr(self, 'logger'):
                    self.logger.warning(
                        f"SUCCESS CHECK [{self.name}/{self.implementation}]: "
                        f"No metrics_history! Using winner['fitness'] → {best_fitness:.6f}"
                    )
            else:
                if hasattr(self, 'logger'):
                    self.logger.error(
                        f"SUCCESS CHECK [{self.name}/{self.implementation}]: "
                        f"No metrics_history AND no winner['fitness']! Using 0.0"
                    )

        # Determine success and log for debugging
        if target_fitness is not None:
            # For adapter implementations, we need to compare in the same scale
            if hasattr(self, 'adapter') and self.adapter is not None and original_target_fitness is not None:
                # Convert best_fitness to normalized scale for comparison
                # Get fitness range from problem for proper normalization
                fitness_range = (0.0, 1.0)  # Default
                if hasattr(problem, 'get_fitness_range'):
                    fitness_range = problem.get_fitness_range()
                normalized_best = self.adapter.normalize_fitness(best_fitness, fitness_range)
                success = normalized_best >= original_target_fitness

                # CRITICAL DEBUG LOGGING - DO NOT REMOVE
                # This helps diagnose fitness normalization and success determination issues
                if hasattr(self, 'logger'):
                    min_fit, max_fit = fitness_range
                    normalization_calc = f"({best_fitness:.6f} - {min_fit:.6f}) / ({max_fit:.6f} - {min_fit:.6f})"
                    expected_result = (best_fitness - min_fit) / (max_fit - min_fit) if max_fit != min_fit else 0.5

                    self.logger.critical(
                        f"\n{'='*80}\n"
                        f"SUCCESS DETERMINATION DEBUG [{self.name}/{self.implementation}]\n"
                        f"{'='*80}\n"
                        f"RAW FITNESS:\n"
                        f"  best_fitness (from metrics) = {best_fitness:.6f}\n"
                        f"\n"
                        f"FITNESS RANGE:\n"
                        f"  problem.get_fitness_range() = {fitness_range}\n"
                        f"  min_fitness = {min_fit:.6f}\n"
                        f"  max_fitness = {max_fit:.6f}\n"
                        f"\n"
                        f"NORMALIZATION CALCULATION:\n"
                        f"  Formula: (raw - min) / (max - min)\n"
                        f"  Substituted: {normalization_calc}\n"
                        f"  Expected result: {expected_result:.6f}\n"
                        f"  Actual normalized_best: {normalized_best:.6f}\n"
                        f"  MATCH: {abs(expected_result - normalized_best) < 0.0001}\n"
                        f"\n"
                        f"THRESHOLD COMPARISON:\n"
                        f"  original_target_fitness (threshold) = {original_target_fitness:.6f}\n"
                        f"  Comparison: {normalized_best:.6f} >= {original_target_fitness:.6f}\n"
                        f"  Result: SUCCESS = {success}\n"
                        f"{'='*80}\n"
                    )

                    # Also log the standard info message
                    self.logger.info(
                        f"SUCCESS CHECK [{self.name}/{self.implementation}] (adapter): "
                        f"best_fitness={best_fitness:.6f} (normalized={normalized_best:.6f}), "
                        f"target_fitness={target_fitness:.6f} (original={original_target_fitness:.6f}), "
                        f"fitness_range={fitness_range}, "
                        f"SUCCESS={success} (comparison: {normalized_best:.6f} >= {original_target_fitness:.6f})"
                    )
            else:
                # Non-adapter implementations or no conversion done
                success = best_fitness >= target_fitness
                if hasattr(self, 'logger'):
                    self.logger.info(
                        f"SUCCESS CHECK [{self.name}/{self.implementation}] (no adapter): "
                        f"best_fitness={best_fitness:.6f}, target_fitness={target_fitness:.6f}, "
                        f"SUCCESS={success} (comparison: {best_fitness:.6f} >= {target_fitness:.6f})"
                    )
        else:
            success = False
            if hasattr(self, 'logger'):
                self.logger.debug(f"No target fitness specified, success=False")

        # Calculate total time from metrics history (use metrics_list to avoid iterator consumption)
        total_time = sum(m.time_elapsed for m in metrics_list) if metrics_list else 0.0

        # Report experiment completion to experiment manager if available
        reporter = get_global_reporter()
        if reporter:
            # Use normalized fitness if available from last metrics
            final_fitness = best_fitness
            if hasattr(self._metrics_storage, 'get_last_metrics'):
                last_metrics = self._metrics_storage.get_last_metrics()
                if last_metrics and hasattr(last_metrics, 'normalized_best_fitness') and last_metrics.normalized_best_fitness is not None:
                    final_fitness = last_metrics.normalized_best_fitness

            # Extract actual fitness value (handle LazyMetricValue if needed)
            if hasattr(final_fitness, 'value'):
                # Check if value is a method or property
                if callable(final_fitness.value):
                    final_fitness = float(final_fitness.value())
                else:
                    final_fitness = float(final_fitness.value)
            elif hasattr(final_fitness, '__float__'):
                final_fitness = float(final_fitness)

            # Report the completion
            reporter.report_completion(
                success=success,
                final_fitness=final_fitness,
                generations=self.current_generation,
                error=None  # No error for successful completion
            )

        return {
            'winner': winner,
            'final_state': state,
            'metrics_history': metrics_list,  # Already converted to list to avoid iterator consumption
            'generations': self.current_generation,
            'network_info': network_info,
            'best_fitness': best_fitness,
            'success': success,
            'total_time': total_time
        }
        
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {
            'name': self.name,
            'implementation': self.implementation,
            'full_name': f"{self.name}_{self.implementation}",
            'supports_gpu': hasattr(self, 'supports_gpu') and self.supports_gpu,
            'supports_checkpoint': hasattr(self, 'supports_checkpoint') and self.supports_checkpoint
        }
    
    def get_best_fitness(self) -> float:
        """Get the best fitness achieved.

        Returns:
            Best fitness value from metrics history
        """
        if self.metrics_history:
            # Convert LazyMetricValue to float before comparison
            # This handles both regular floats and LazyMetricValue objects
            def _to_float(value):
                if hasattr(value, 'value'):
                    # LazyMetricValue - check if value is callable
                    if callable(value.value):
                        return float(value.value())
                    else:
                        return float(value.value)
                elif hasattr(value, '__float__'):
                    return float(value)
                else:
                    return float(value)

            return max(_to_float(m.best_fitness) for m in self.metrics_history)
        return 0.0
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final statistics after evolution.
        
        Returns:
            Dictionary containing final evolution statistics
        """
        stats = {
            'best_fitness': self.get_best_fitness(),
            'generations': self.current_generation,
            'total_time': sum(m.time_elapsed for m in self.metrics_history) if self.metrics_history else 0.0,
        }
        
        # Add final network info if available
        if self.metrics_history:
            last_metrics = self.metrics_history[-1]
            stats['final_species'] = last_metrics.num_species
            stats['final_evaluations'] = last_metrics.evaluations
            
        return stats
    
    def get_config_metadata(self) -> Optional[Dict[str, Any]]:
        """Get configuration metadata from ConfigManager if available.
        
        Returns:
            Dictionary containing configuration metadata or None if not available
        """
        # Default implementation returns None
        # Implementations should override this if they have metadata
        return getattr(self, '_config_metadata', None)
    
    def get_implementation_metadata(self) -> Dict[str, Any]:
        """Get implementation-specific metadata for system configuration display.
        
        Returns:
            Dictionary containing implementation metadata
        """
        # Default implementation returns basic information
        return {
            'display_name': self.implementation.replace('_', '-').title(),
            'version': 'Unknown',
            'backend': 'Unknown',
            'gpu_support': False,
            'jit_support': False,
            'performance': {'speed': 'Unknown', 'memory': 'Unknown'},
            'status': 'Available'
        }