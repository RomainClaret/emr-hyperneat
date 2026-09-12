"""
NEAT Adapter Base Class and Implementation.

This module provides the adapter layer that sits between NEAT implementations
and problems, handling all format conversions and making problems truly
implementation-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple, Optional
import numpy as np
import os
from dataclasses import dataclass, field


@dataclass
class NetworkInfo:
    """Standardized network information across implementations."""
    num_nodes: int
    num_connections: int
    num_enabled_connections: int
    num_inputs: int
    num_outputs: int
    num_hidden: int
    connections: List[Tuple[int, int, float]]  # (from, to, weight)
    nodes: Dict[int, Dict[str, Any]]  # node_id -> properties


@dataclass
class EvolutionMetrics:
    """Standardized evolution metrics across implementations."""
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
    stagnation_rate: float = 0.0
    innovation_rate: float = 0.0
    species_turnover_rate: float = 0.0
    diversity_maintenance: float = 0.0


@dataclass
class PerformanceMetrics:
    """Implementation-specific performance metrics."""
    total_time: float
    time_per_generation: float
    time_per_evaluation: float
    overhead_metrics: Dict[str, float] = field(default_factory=dict)  # e.g., JIT compilation time
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0


class NEATAdapter(ABC):
    """Abstract base class for NEAT implementation adapters.
    
    Adapters handle all the implementation-specific details like:
    - Input format (bias handling)
    - Network activation methods
    - Fitness scale conversions
    - Metric normalization
    
    This allows problems to work with any NEAT implementation without
    knowing its specific requirements.
    """
    
    def __init__(self, name: str):
        """Initialize adapter.
        
        Args:
            name: Name of the implementation (e.g., 'pureples', 'tensorneat')
        """
        self.name = name
        self._dtype = np.float32  # Default to float32 for performance
        self._precision = 'float32'
    
    def set_precision(self, precision: str):
        """Set the precision for this adapter.
        
        Note: Not all implementations support precision changes. For example,
        PUREPLES uses standard Python/NumPy and should keep its default behavior.
        This is primarily for JAX-based implementations like TensorNEAT.
        
        Args:
            precision: Either 'float32' or 'float64'
            
        Raises:
            ValueError: If precision is not valid
        """
        if precision not in ['float32', 'float64']:
            raise ValueError(f"Invalid precision: {precision}. Must be 'float32' or 'float64'")
        
        self._precision = precision
        self._dtype = np.float32 if precision == 'float32' else np.float64
    
    def supports_precision_control(self) -> bool:
        """Check if this adapter supports precision control.
        
        Returns:
            True if precision can be configured
        """
        # By default, assume no precision control
        # Subclasses should override if they support it
        return False
    
    @property
    def dtype(self):
        """Get the numpy dtype for this adapter."""
        return self._dtype
    
    @property
    def precision(self):
        """Get the precision setting for this adapter."""
        return self._precision
    
    # Input Preparation Methods
    
    @abstractmethod
    def prepare_input(self, raw_input: np.ndarray) -> Any:
        """Prepare a single input for the implementation.
        
        Args:
            raw_input: Raw input array from the problem
            
        Returns:
            Prepared input in implementation-specific format
        """
        pass
    
    @abstractmethod
    def prepare_batch(self, inputs: List[np.ndarray]) -> Any:
        """Prepare a batch of inputs for the implementation.
        
        Args:
            inputs: List of raw input arrays
            
        Returns:
            Prepared batch in implementation-specific format
        """
        pass
    
    @abstractmethod
    def needs_bias(self) -> bool:
        """Check if this implementation needs bias in inputs.
        
        Returns:
            True if bias should be added to inputs
        """
        pass
    
    # Network Activation Methods
    
    @abstractmethod
    def activate(self, network: Any, prepared_input: Any) -> np.ndarray:
        """Activate network with a single prepared input.
        
        Args:
            network: Network object from the implementation
            prepared_input: Input prepared by prepare_input()
            
        Returns:
            Network output as numpy array
        """
        pass
    
    @abstractmethod
    def activate_batch(self, network: Any, prepared_batch: Any) -> np.ndarray:
        """Activate network with a batch of prepared inputs.
        
        Args:
            network: Network object from the implementation
            prepared_batch: Batch prepared by prepare_batch()
            
        Returns:
            Array of network outputs
        """
        pass
    
    # Fitness Conversion Methods
    
    @abstractmethod
    def normalize_fitness(self, raw_fitness: float, 
                         fitness_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Convert implementation-specific fitness to normalized [0, 1] range.
        
        Args:
            raw_fitness: Fitness value from the implementation
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Normalized fitness in [0, 1] range
        """
        pass
    
    @abstractmethod
    def denormalize_fitness(self, normalized_fitness: float,
                           fitness_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Convert normalized [0, 1] fitness to implementation-specific scale.
        
        Args:
            normalized_fitness: Fitness in [0, 1] range
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Fitness in implementation-specific scale
        """
        pass
    
    @abstractmethod
    def get_fitness_threshold(self, normalized_threshold: float,
                             fitness_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Convert normalized threshold to implementation-specific threshold.
        
        Args:
            normalized_threshold: Threshold in [0, 1] range
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Threshold in implementation-specific scale
        """
        pass
    
    # Metric Conversion Methods
    
    def convert_metrics(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        """Convert raw metrics to normalized values.
        
        Default implementation just passes through metrics.
        Override for implementation-specific conversions.
        
        Args:
            raw_metrics: Dictionary of raw metric values
            
        Returns:
            Dictionary of normalized metric values
        """
        return raw_metrics.copy()
    
    # Utility Methods
    
    def is_discrete_output(self, output: np.ndarray) -> bool:
        """Check if network output represents discrete classes.
        
        Args:
            output: Network output array
            
        Returns:
            True if output should be treated as class probabilities
        """
        # Heuristic: if output has more than one element, assume classification
        return len(output) > 1
    
    def get_predicted_class(self, output: np.ndarray) -> int:
        """Get predicted class from network output.
        
        Args:
            output: Network output array
            
        Returns:
            Index of predicted class
        """
        return int(np.argmax(output))
    
    def threshold_output(self, output: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Apply threshold to continuous output.
        
        Args:
            output: Network output array
            threshold: Threshold value
            
        Returns:
            Thresholded output (0 or 1)
        """
        return (output > threshold).astype(np.float32)
    
    # Problem Evaluation Helper
    
    def evaluate_network(self, 
                        network: Any, 
                        inputs: List[np.ndarray], 
                        targets: List[np.ndarray],
                        calculate_metrics: Optional[Any] = None,
                        return_raw: bool = False,
                        problem_requirements: Optional[Dict[str, Any]] = None) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
        """Helper method to evaluate network on a dataset.
        
        Args:
            network: Network to evaluate
            inputs: List of input arrays
            targets: List of target arrays
            calculate_metrics: Optional function to calculate metrics
            return_raw: Unused, kept for compatibility
            problem_requirements: Optional problem requirements for transformation
            
        Returns:
            Tuple of (normalized_fitness, details_dict)
        """
        predictions = []
        
        # Process each input
        for inp in inputs:
            # Transform input if requirements provided
            if problem_requirements:
                inp = self.transform_single_input(inp, problem_requirements)
            prepared = self.prepare_input(inp)
            output = self.activate(network, prepared)
            predictions.append(output)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Handle shape compatibility for single-output problems
        if targets.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] == 1:
            # Squeeze the last dimension for single-output problems
            predictions = predictions.squeeze(-1)
        
        # Calculate metrics using provided function or default
        if calculate_metrics is not None:
            metrics = calculate_metrics(predictions, targets)
        else:
            metrics = self._calculate_metrics(predictions, targets)
        
        # Calculate fitness (normalized)
        # For classification problems, use accuracy
        # For regression problems, use 1 - MSE (clamped to [0, 1])
        if 'accuracy' in metrics:
            fitness = metrics['accuracy']
        elif 'mse' in metrics:
            fitness = max(0.0, 1.0 - metrics['mse'])
        else:
            fitness = 0.0
            
        # Return in expected format: (fitness, details)
        details = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
        
        return fitness, details
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate standard metrics from predictions and targets.
        
        Args:
            predictions: Array of predictions
            targets: Array of targets
            
        Returns:
            Dictionary of metric values
        """
        # Mean squared error
        mse = np.mean((predictions - targets) ** 2)
        
        # Sum of squared errors
        sse = np.sum((predictions - targets) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(predictions - targets))
        
        # For classification tasks
        if self.is_discrete_output(predictions[0]):
            pred_classes = np.array([self.get_predicted_class(p) for p in predictions])
            true_classes = np.array([self.get_predicted_class(t) for t in targets])
            accuracy = np.mean(pred_classes == true_classes)
        else:
            # For binary outputs, use threshold
            pred_binary = self.threshold_output(predictions.flatten())
            true_binary = self.threshold_output(targets.flatten())
            accuracy = np.mean(pred_binary == true_binary)
        
        return {
            'mse': float(mse),
            'sse': float(sse),
            'mae': float(mae),
            'accuracy': float(accuracy),
            'rmse': float(np.sqrt(mse))
        }
    
    # New abstract methods for enhanced adapter functionality
    
    @abstractmethod
    def get_implementation_type(self) -> str:
        """Get the implementation type identifier.
        
        Returns:
            Implementation type string (e.g., 'pureples', 'tensorneat')
        """
        pass
    
    @abstractmethod
    def normalize_fitness_with_context(self, raw_fitness: float, problem: str = None) -> float:
        """Convert implementation-specific fitness to normalized [0, 1] range with problem context.
        
        This is an enhanced version of normalize_fitness that can consider the problem
        type when normalizing fitness values.
        
        Args:
            raw_fitness: Fitness value from the implementation
            problem: Optional problem name for context-aware normalization
            
        Returns:
            Normalized fitness in [0, 1] range
        """
        pass
    
    @abstractmethod
    def extract_network_info(self, result: Any) -> NetworkInfo:
        """Extract standardized network information from implementation-specific result.
        
        Args:
            result: Result object from the implementation
            
        Returns:
            Standardized NetworkInfo object
        """
        pass
    
    @abstractmethod
    def extract_evolution_metrics(self, result: Any, generation: int = None) -> EvolutionMetrics:
        """Extract standardized evolution metrics from implementation-specific result.
        
        Args:
            result: Result object from the implementation
            generation: Optional generation number if not in result
            
        Returns:
            Standardized EvolutionMetrics object
        """
        pass
    
    @abstractmethod
    def calculate_performance_overhead(self, result: Any) -> Dict[str, float]:
        """Calculate implementation-specific performance overhead.
        
        For example:
        - TensorNEAT: JIT compilation time
        - PUREPLES: Any specific overhead
        
        Args:
            result: Result object from the implementation
            
        Returns:
            Dictionary of overhead metrics (can be empty)
        """
        pass
    
    def extract_all_metrics(self, result: Any) -> Dict[str, Any]:
        """Extract all available metrics from result.
        
        This is a convenience method that calls all extraction methods.
        
        Args:
            result: Result object from the implementation
            
        Returns:
            Dictionary containing all extracted metrics
        """
        network_info = self.extract_network_info(result)
        evolution_metrics = self.extract_evolution_metrics(result)
        performance_overhead = self.calculate_performance_overhead(result)
        
        return {
            'network': network_info,
            'evolution': evolution_metrics,
            'overhead': performance_overhead
        }
    
    # Parameter Mapping Methods
    
    @abstractmethod
    def get_parameter_mapping(self) -> Dict[str, str]:
        """Get parameter name mapping from unified names to implementation-specific names.
        
        This allows the framework to use consistent parameter names while each
        implementation can use its own naming conventions.
        
        Returns:
            Dictionary mapping unified parameter names to implementation names
        """
        pass
    
    @abstractmethod
    def to_native_config(self, unified_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified configuration to implementation-specific format.
        
        Args:
            unified_config: Configuration using unified parameter names
            
        Returns:
            Configuration using implementation-specific parameter names
        """
        pass
    
    @abstractmethod
    def from_native_config(self, native_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert implementation-specific configuration to unified format.
        
        Args:
            native_config: Configuration using implementation-specific names
            
        Returns:
            Configuration using unified parameter names
        """
        pass
    
    # Hardware and Parallelization Methods
    
    @abstractmethod
    def get_hardware_requirements(self) -> Dict[str, Any]:
        """Get hardware requirements and preferences for this implementation.
        
        Returns:
            Dictionary with keys:
            - 'gpu_required': bool
            - 'gpu_recommended': bool
            - 'min_memory_mb': int
            - 'recommended_memory_mb': int
            - 'min_population_size': int
            - 'recommended_population_size': int
        """
        pass
    
    @abstractmethod
    def supports_multiprocessing(self) -> bool:
        """Check if this implementation supports multiprocessing.
        
        Returns:
            True if multiprocessing is supported
        """
        pass
    
    @abstractmethod
    def get_optimal_parallelization(self, population_size: int, 
                                  num_cpus: int = None, 
                                  has_gpu: bool = False) -> Dict[str, Any]:
        """Get optimal parallelization configuration for given resources.
        
        Args:
            population_size: Size of the population
            num_cpus: Number of available CPUs
            has_gpu: Whether GPU is available
            
        Returns:
            Dictionary with keys:
            - 'use_multiprocessing': bool
            - 'n_processes': int
            - 'batch_size': int (for GPU batching)
            - 'notes': str (explanation of choices)
        """
        pass
    
    # Problem Transformation Methods
    
    @abstractmethod
    def transform_problem(self, problem: 'BaseProblem') -> 'BaseProblem':
        """Transform a problem to work with this implementation.
        
        This handles implementation-specific requirements like bias handling.
        Instead of problems knowing about implementations, adapters transform
        problems as needed.
        
        Args:
            problem: Original problem instance
            
        Returns:
            Transformed problem instance (may be the same object)
        """
        pass
    
    # Visualization Methods
    
    def get_visualization_renderer(self) -> Optional['NetworkRenderer']:
        """Get custom visualization renderer for this implementation.
        
        Returns:
            Custom renderer or None to use default visualization
        """
        return None
    
    # Capability Declaration Methods
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Declare what this implementation supports.
        
        Returns:
            Dictionary with capability flags:
            - 'batch_evaluation': bool
            - 'gpu_acceleration': bool
            - 'distributed_evaluation': bool
            - 'dynamic_population': bool
            - 'checkpointing': bool
            - 'real_time_visualization': bool
            - 'custom_mutations': bool
            - 'substrate_evolution': bool (for HyperNEAT variants)
        """
        pass
    
    @abstractmethod
    def get_limitations(self) -> List[str]:
        """Get list of known limitations for this implementation.
        
        Returns:
            List of limitation descriptions (e.g., "Requires population >= 300")
        """
        pass
    
    # Evolution Dynamics Extraction
    
    @abstractmethod
    def extract_species_count(self, result: Any) -> int:
        """Extract current species count from result.
        
        Args:
            result: Result object from implementation
            
        Returns:
            Number of species
        """
        pass
    
    @abstractmethod
    def extract_species_history(self, result: Any) -> List[int]:
        """Extract species count history from result.
        
        Args:
            result: Result object from implementation
            
        Returns:
            List of species counts per generation
        """
        pass
    
    # Success Criteria Methods
    
    @abstractmethod
    def get_success_criteria(self, problem: str) -> Dict[str, Any]:
        """Get implementation-specific success criteria for a problem.
        
        Args:
            problem: Problem name
            
        Returns:
            Dictionary with:
            - 'fitness_threshold': float (in implementation scale)
            - 'min_generations': int (optional)
            - 'max_stagnation': int (optional)
        """
        pass
    
    @abstractmethod
    def is_success(self, fitness: float, problem: str) -> bool:
        """Check if a fitness value represents success for a problem.
        
        Args:
            fitness: Raw fitness value from implementation
            problem: Problem name
            
        Returns:
            True if fitness meets success criteria
        """
        pass
    
    @abstractmethod
    def get_default_config(self, problem: str, population_size: int = None) -> Dict[str, Any]:
        """Get default configuration for a problem.
        
        Args:
            problem: Problem name
            population_size: Optional population size to tailor config
            
        Returns:
            Default configuration dictionary in native format
        """
        pass
    
    # Data Transformation Methods
    
    def transform_problem_data(self, 
                             data: List[Tuple[np.ndarray, np.ndarray]], 
                             requirements: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Transform problem data based on implementation needs.
        
        This method handles the transformation of problem data to match
        implementation requirements, particularly for bias handling.
        
        Args:
            data: List of (input, output) tuples from problem
            requirements: Problem requirements from get_problem_requirements()
            
        Returns:
            Transformed list of (input, output) tuples
        """
        provides_bias = requirements.get('provides_bias', False)
        needs_bias = self.needs_bias()
        
        if needs_bias and not provides_bias:
            # Add bias column to inputs
            transformed_data = []
            for inp, out in data:
                # Add 1.0 as bias to the end of input
                inp_with_bias = np.append(inp, 1.0)
                transformed_data.append((inp_with_bias, out))
            return transformed_data
            
        elif not needs_bias and provides_bias:
            # Remove bias column from inputs (assume it's the last column)
            transformed_data = []
            for inp, out in data:
                # Remove last column (bias)
                inp_without_bias = inp[:-1]
                transformed_data.append((inp_without_bias, out))
            return transformed_data
            
        else:
            # No transformation needed
            return data
    
    def transform_single_input(self, 
                             input_array: np.ndarray, 
                             requirements: Dict[str, Any]) -> np.ndarray:
        """Transform a single input based on implementation needs.
        
        Args:
            input_array: Single input array
            requirements: Problem requirements
            
        Returns:
            Transformed input array
        """
        provides_bias = requirements.get('provides_bias', False)
        needs_bias = self.needs_bias()
        
        if needs_bias and not provides_bias:
            # Add bias
            return np.append(input_array, 1.0)
        elif not needs_bias and provides_bias:
            # Remove bias (last element)
            return input_array[:-1]
        else:
            # No transformation
            return input_array
    
    def get_transformed_input_size(self, 
                                 original_size: int, 
                                 requirements: Dict[str, Any]) -> int:
        """Get the input size after transformation.
        
        Args:
            original_size: Original input size from problem
            requirements: Problem requirements
            
        Returns:
            Input size after transformation
        """
        provides_bias = requirements.get('provides_bias', False)
        needs_bias = self.needs_bias()
        
        if needs_bias and not provides_bias:
            # Will add bias
            return original_size + 1
        elif not needs_bias and provides_bias:
            # Will remove bias
            return original_size - 1
        else:
            # No change
            return original_size
    
    def __repr__(self) -> str:
        """String representation of adapter."""
        return f"{self.__class__.__name__}(name='{self.name}')"