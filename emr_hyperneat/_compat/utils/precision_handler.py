"""
Precision configuration handler for the experiment framework.

Provides centralized precision configuration logic with automatic selection,
validation, and implementation-specific handling.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PrecisionHandler:
    """Handles precision configuration for different implementations."""
    
    # Implementation precision support matrix
    PRECISION_SUPPORT = {
        'pureples': {
            'supports_precision_control': False,
            'default_precision': 'float64',  # NumPy default
            'enforced_precision': None,
            'notes': 'Uses standard NumPy behavior'
        },
        'tensorneat': {
            'supports_precision_control': True,
            'default_precision': 'float32',
            'enforced_precision': None,
            'notes': 'Uses JAX backend, supports both float32 and float64 based on JAX_ENABLE_X64 setting'
        }
    }
    
    @staticmethod
    def get_auto_precision(has_gpu: bool = False, 
                          problem_type: str = 'classification',
                          population_size: int = 150) -> str:
        """
        Automatically select precision based on hardware and problem characteristics.
        
        Args:
            has_gpu: Whether GPU is available
            problem_type: Type of problem ('classification' or 'regression')
            population_size: Size of the population
            
        Returns:
            Recommended precision ('float32' or 'float64')
        """
        # GPU strongly prefers float32 (up to 30x faster)
        if has_gpu:
            return 'float32'
        
        # CPU can handle float64 with smaller performance penalty
        # Use float64 for scientific computing or when precision matters
        if problem_type == 'regression' or population_size < 200:
            return 'float64'
        
        # Default to float32 for performance
        return 'float32'
    
    @staticmethod
    def configure_precision(implementation: str, 
                           requested_precision: Optional[str] = None,
                           has_gpu: bool = False,
                           problem_type: str = 'classification',
                           population_size: int = 150) -> Tuple[str, str, bool]:
        """
        Configure precision for a specific implementation.
        
        Args:
            implementation: Implementation name ('pureples', 'tensorneat')
            requested_precision: User-requested precision ('float32', 'float64', 'auto', None)
            has_gpu: Whether GPU is available
            problem_type: Type of problem
            population_size: Size of the population
            
        Returns:
            Tuple of (actual_precision, requested_precision, precision_changed)
        """
        # Get implementation info
        impl_info = PrecisionHandler.PRECISION_SUPPORT.get(implementation, {})
        
        # Handle auto or None precision
        if requested_precision == 'auto' or requested_precision is None:
            requested_precision = PrecisionHandler.get_auto_precision(
                has_gpu, problem_type, population_size
            )
        
        # Validate requested precision
        if requested_precision not in ['float32', 'float64']:
            logger.warning(f"Invalid precision '{requested_precision}', defaulting to float32")
            requested_precision = 'float32'
        
        # Check if implementation has enforced precision
        if impl_info.get('enforced_precision'):
            actual_precision = impl_info['enforced_precision']
            precision_changed = actual_precision != requested_precision
            
            if precision_changed:
                logger.warning(
                    f"{implementation} enforces {actual_precision} precision. "
                    f"Requested {requested_precision} will be ignored. "
                    f"Reason: {impl_info.get('notes', 'Implementation constraint')}"
                )
        else:
            # Implementation can use requested precision (or its default)
            actual_precision = requested_precision
            precision_changed = False
        
        return actual_precision, requested_precision, precision_changed
    
    @staticmethod
    def get_numpy_dtype(precision: str) -> np.dtype:
        """
        Get NumPy dtype for a precision string.
        
        Args:
            precision: Precision string ('float32' or 'float64')
            
        Returns:
            NumPy dtype
        """
        if precision == 'float32':
            return np.float32
        elif precision == 'float64':
            return np.float64
        else:
            raise ValueError(f"Invalid precision: {precision}")
    
    @staticmethod
    def configure_jax_precision(precision: str):
        """
        Configure JAX precision if not already set.
        
        Args:
            precision: Desired precision ('float32' or 'float64')
        """
        # Only configure if not already set
        if 'JAX_ENABLE_X64' not in os.environ:
            if precision == 'float64':
                os.environ['JAX_ENABLE_X64'] = 'true'
                logger.info("Configured JAX for float64 precision")
            else:
                os.environ['JAX_ENABLE_X64'] = '0'
                logger.info("Configured JAX for float32 precision")
        else:
            current = os.environ['JAX_ENABLE_X64']
            expected = 'true' if precision == 'float64' else '0'
            if current != expected:
                logger.warning(
                    f"JAX precision already configured (JAX_ENABLE_X64={current}). "
                    f"Cannot change to {precision}."
                )
    
    @staticmethod
    def get_precision_info(implementation: str, precision: str) -> Dict[str, Any]:
        """
        Get detailed information about precision configuration.
        
        Args:
            implementation: Implementation name
            precision: Precision setting
            
        Returns:
            Dictionary with precision information
        """
        impl_info = PrecisionHandler.PRECISION_SUPPORT.get(implementation, {})
        
        return {
            'requested_precision': precision,
            'actual_precision': impl_info.get('enforced_precision', precision),
            'supports_control': impl_info.get('supports_precision_control', False),
            'default_precision': impl_info.get('default_precision', 'float32'),
            'notes': impl_info.get('notes', ''),
            'memory_multiplier': 2.0 if precision == 'float64' else 1.0,
            'gpu_performance_impact': '~30x slower' if precision == 'float64' else 'optimal'
        }
    
    @staticmethod
    def format_precision_warning(implementation: str, 
                                requested: str, 
                                actual: str) -> str:
        """
        Format a user-friendly warning about precision mismatch.
        
        Args:
            implementation: Implementation name
            requested: Requested precision
            actual: Actual precision that will be used
            
        Returns:
            Formatted warning message
        """
        impl_info = PrecisionHandler.PRECISION_SUPPORT.get(implementation, {})
        
        return (
            f"⚠️  Precision Configuration Warning\n"
            f"   Implementation: {implementation}\n"
            f"   Requested: {requested}\n"
            f"   Actual: {actual}\n"
            f"   Reason: {impl_info.get('notes', 'Implementation constraint')}\n"
            f"   Impact: This will not affect results but may impact performance expectations."
        )
    
    @staticmethod
    def get_performance_implications(precision: str, has_gpu: bool) -> Dict[str, str]:
        """
        Get performance implications of precision choice.
        
        Args:
            precision: Precision setting
            has_gpu: Whether GPU is available
            
        Returns:
            Dictionary with performance implications
        """
        if precision == 'float64':
            if has_gpu:
                return {
                    'speed': '~30x slower than float32',
                    'memory': '2x memory usage',
                    'recommendation': 'Use float32 for GPU unless precision is critical'
                }
            else:
                return {
                    'speed': '~2x slower than float32',
                    'memory': '2x memory usage',
                    'recommendation': 'Acceptable for CPU, consider float32 for large populations'
                }
        else:  # float32
            return {
                'speed': 'Optimal performance',
                'memory': 'Standard memory usage',
                'recommendation': 'Recommended for most neural network tasks'
            }