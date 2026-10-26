"""
Precision configuration handler for the experiment framework.

Provides centralized precision configuration logic with automatic selection,
validation, and implementation-specific handling.
"""

import logging
from typing import Optional, Tuple

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
            f"Precision Configuration Warning\n"
            f"   Implementation: {implementation}\n"
            f"   Requested: {requested}\n"
            f"   Actual: {actual}\n"
            f"   Reason: {impl_info.get('notes', 'Implementation constraint')}\n"
            f"   Impact: This will not affect results but may impact performance expectations."
        )
