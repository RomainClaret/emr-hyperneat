"""Utility functions for handling metrics in the experiment framework."""

from typing import Any, Dict
from dataclasses import asdict, is_dataclass


def ensure_dict(obj: Any) -> Dict[str, Any]:
    """
    Ensure an object is converted to a dictionary.
    
    Handles:
    - Dictionaries (returned as-is)
    - Objects with to_dict() method
    - Dataclasses
    - Objects with __dict__ attribute
    - None (returns empty dict)
    
    Args:
        obj: Any object to convert
        
    Returns:
        Dictionary representation
    """
    if obj is None:
        return {}
    
    if isinstance(obj, dict):
        return obj
    
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    
    if is_dataclass(obj):
        return asdict(obj)
    
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    
    # Last resort - try to convert to dict
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {}
