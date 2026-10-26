"""
NEAT Adapter Base Class and Implementation.

This module provides the adapter layer that sits between NEAT implementations
and problems, handling all format conversions and making problems truly
implementation-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


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
    
    @property
    def dtype(self):
        """Get the numpy dtype for this adapter."""
        return self._dtype
    
    @property
    def precision(self):
        """Get the precision setting for this adapter."""
        return self._precision
    
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
    
    # Implementation Type, Normalization and Metrics Methods
    
    @abstractmethod
    def extract_network_info(self, result: Any) -> NetworkInfo:
        """Extract standardized network information from implementation-specific result.
        
        Args:
            result: Result object from the implementation
            
        Returns:
            Standardized NetworkInfo object
        """
        pass
    
    # Data Transformation Methods
    
    def __repr__(self) -> str:
        """String representation of adapter."""
        return f"{self.__class__.__name__}(name='{self.name}')"