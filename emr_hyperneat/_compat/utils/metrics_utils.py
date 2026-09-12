"""Utility functions for handling metrics in the experiment framework."""

from typing import Any, Dict, Union, List
from dataclasses import asdict, is_dataclass


def get_metric_value(metric_obj: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """
    Safely get a metric value from either a dict or an AlgorithmMetrics object.
    
    Args:
        metric_obj: Either a dictionary or an AlgorithmMetrics object
        key: The metric key/attribute to retrieve
        default: Default value if key/attribute not found
        
    Returns:
        The metric value or default
    """
    if isinstance(metric_obj, dict):
        return metric_obj.get(key, default)
    elif hasattr(metric_obj, key):
        return getattr(metric_obj, key, default)
    else:
        return default


def process_metrics_history(metrics_history: List[Union[Dict[str, Any], Any]]) -> List[Dict[str, Any]]:
    """
    Convert a metrics history list to ensure all entries are dictionaries.
    
    Args:
        metrics_history: List of metrics (can be dicts or AlgorithmMetrics objects)
        
    Returns:
        List of dictionaries
    """
    processed = []
    for metric in metrics_history:
        if isinstance(metric, dict):
            processed.append(metric)
        elif hasattr(metric, 'to_dict'):
            processed.append(metric.to_dict())
        else:
            # Try to convert to dict using dataclass asdict
            try:
                from dataclasses import asdict
                processed.append(asdict(metric))
            except Exception:
                # If all else fails, extract known attributes
                metric_dict = {}
                for attr in ['generation', 'best_fitness', 'mean_fitness', 
                           'min_fitness', 'max_fitness', 'std_fitness',
                           'num_species', 'evaluations', 'time_elapsed',
                           'custom_metrics']:
                    if hasattr(metric, attr):
                        metric_dict[attr] = getattr(metric, attr)
                processed.append(metric_dict)
    return processed


def extract_fitness_progression(metrics_history: List[Union[Dict[str, Any], Any]], 
                               limit: int = None) -> List[float]:
    """
    Extract fitness progression from metrics history.
    
    Args:
        metrics_history: List of metrics (can be dicts or AlgorithmMetrics objects)
        limit: Maximum number of generations to include (None for all)
        
    Returns:
        List of best fitness values
    """
    progression = []
    history_subset = metrics_history[:limit] if limit else metrics_history
    
    for metric in history_subset:
        fitness = get_metric_value(metric, 'best_fitness', float('nan'))
        progression.append(fitness)
    
    return progression


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


def safe_get_metric(obj: Any, metric_name: str, default: Any = None) -> Any:
    """
    Safely get a metric value from an object that could be a dict or dataclass.
    
    Args:
        obj: Object to get metric from
        metric_name: Name of the metric
        default: Default value if metric not found
        
    Returns:
        Metric value or default
    """
    if isinstance(obj, dict):
        return obj.get(metric_name, default)
    elif hasattr(obj, metric_name):
        return getattr(obj, metric_name, default)
    else:
        return default