"""
TensorNEAT Implementation Adapter.

This adapter handles all TensorNEAT-specific conversions and requirements.
"""

from typing import Any, Dict
import numpy as np
import logging

from .neat_adapter import NEATAdapter


class TensorNEATAdapter(NEATAdapter):
    """Adapter for TensorNEAT implementation.
    
    TensorNEAT characteristics:
    - Does not require bias input (bias is built into nodes)
    - Returns fitness = -mean_square_error (negative MSE)
    - Networks are JAX functions that work with vmap
    - Uses JAX arrays for computation
    """
    
    def __init__(self):
        """Initialize TensorNEAT adapter."""
        super().__init__('tensorneat')
        
        # Delay JAX import until first use to allow precision configuration
        self._jax = None
        self._jnp = None
        self._dtype = None
        self._jax_initialized = False
    
    def _ensure_jax_initialized(self):
        """Initialize JAX if not already done."""
        if not self._jax_initialized:
            # Import JAX using smart configuration
            try:
                from emr_hyperneat._compat.utils.jax_smart_config import configure_jax_smart, silent_jax_import
                # Configure JAX if not already configured
                configure_jax_smart()
            except ImportError:
                # Fallback if jax_smart_config not available
                try:
                    # Try old CPU-only config
                    from emr_hyperneat._compat.utils.jax_cpu_config import silent_jax_import
                except ImportError:
                    # Final fallback - basic configuration
                    import os
                    if 'JAX_PLATFORM_NAME' not in os.environ:
                        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
                    import jax
                    import jax.numpy as jnp
                    silent_jax_import = lambda: (jax, jnp)
            
            # Import JAX silently (suppresses Metal warnings)
            self._jax, self._jnp = silent_jax_import()
            
            # Use the appropriate dtype based on JAX configuration
            # float32 is recommended for GPU performance (float64 is ~30x slower on GPU)
            # The dtype will be determined by JAX's configuration (JAX_ENABLE_X64 environment variable)
            self._dtype = self._jnp.float32 if not self._jax.config.x64_enabled else self._jnp.float64
            self._jax_initialized = True
    
    # Network Activation Methods
    
    def activate(self, network: Any, prepared_input: Any) -> np.ndarray:
        """Activate TensorNEAT network.
        
        Args:
            network: TensorNEAT network function
            prepared_input: JAX array from prepare_input()
            
        Returns:
            Network output as numpy array
        """
        self._ensure_jax_initialized()
        logger = logging.getLogger(__name__)
        
        try:
            # Check if input is already batched
            if prepared_input.ndim == 2:
                # It's a batch - use JAX vmap for vectorized execution
                # This is much more efficient than Python loops
                network_vmap = self._jax.vmap(network)
                outputs = network_vmap(prepared_input)
                result = np.array(outputs)
            else:
                # Single input - apply directly
                output = network(prepared_input)
                result = np.array(output)
            
            # Check the outputs for NaN and log the common causes
            if np.any(np.isnan(result)):
                logger.warning(
                    "TensorNEAT network produced NaN outputs - likely disconnected network.\n"
                    "Common causes:\n"
                    "1. Network started with init_hidden_layers=[] (no connections)\n"
                    "2. All connections were deleted through mutation\n"
                    "3. Numerical instability in activation functions\n"
                    "Fix: Set init_hidden_layers=[0] in config for initial connections"
                )
                # Log additional debug info if available
                if hasattr(network, '__name__'):
                    logger.debug(f"Network function: {network.__name__}")
            
            # Ensure proper output shape
            if result.ndim == 0:
                result = np.array([result])
            elif result.ndim > 1 and result.shape[0] == 1:
                # Remove unnecessary batch dimension for single inputs
                result = result.squeeze(0)
            
            # Return with the appropriate dtype based on JAX configuration
            dtype_str = 'float32' if self._dtype == self._jnp.float32 else 'float64'
            return result.astype(getattr(np, dtype_str))
            
        except Exception as e:
            logger.error(f"TensorNEAT network activation failed: {e}")
            # Return NaN array to indicate failure
            if prepared_input.ndim == 2:
                # Batch input
                return np.full((prepared_input.shape[0], 1), np.nan)
            else:
                # Single input
                return np.array([np.nan])
    
    # Fitness Conversion Methods
    
    def normalize_fitness(self, raw_fitness: float, 
                         fitness_range: tuple[float, float] = (-1.0, 0.0)) -> float:
        """Convert TensorNEAT fitness to normalized [0, 1] range.
        
        Uses the provided fitness range from the problem to properly scale
        the raw fitness value to the standard [0, 1] normalized range.
        TensorNEAT typically uses negative fitness (-MSE), where 0 is perfect.
        
        Args:
            raw_fitness: TensorNEAT fitness value
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Normalized fitness in [0, 1]
        """
        min_val, max_val = fitness_range
        
        # Handle LazyMetricValue (convert to float first)
        if hasattr(raw_fitness, '__float__'):
            raw_fitness = float(raw_fitness)
        
        # Handle NaN/Inf consistently
        if np.isnan(raw_fitness):
            logger = logging.getLogger(__name__)
            logger.warning("TensorNEAT produced NaN fitness in normalize_fitness")
            return 0.0
        
        if np.isinf(raw_fitness):
            logger = logging.getLogger(__name__)
            logger.warning(f"TensorNEAT produced infinite fitness in normalize_fitness: {raw_fitness}")
            return 0.0
        
        # Handle error marker (TensorNEAT specific)
        if raw_fitness <= -1000.0:
            return 0.0
        
        # Handle edge case where min == max
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        
        # Scale from [min_val, max_val] to [0, 1]
        normalized = (raw_fitness - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def denormalize_fitness(self, normalized_fitness: float,
                           fitness_range: tuple[float, float] = (-1.0, 0.0)) -> float:
        """Convert normalized fitness to TensorNEAT scale.
        
        Uses the provided fitness range to convert from the normalized [0, 1]
        range back to the problem's raw fitness scale.
        
        Args:
            normalized_fitness: Fitness in [0, 1] range
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Fitness in TensorNEAT's scale
        """
        min_val, max_val = fitness_range
        
        # Ensure input is in valid range
        normalized_fitness = max(0.0, min(1.0, normalized_fitness))
        
        # Scale from [0, 1] to [min_val, max_val]
        return min_val + normalized_fitness * (max_val - min_val)
    
    def get_fitness_threshold(self, normalized_threshold: float,
                             fitness_range: tuple[float, float] = (-1.0, 0.0)) -> float:
        """Convert normalized threshold to TensorNEAT scale.
        
        Args:
            normalized_threshold: Threshold in [0, 1] range
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Threshold in TensorNEAT's scale
        """
        # Use denormalize_fitness for consistency
        return self.denormalize_fitness(normalized_threshold, fitness_range)
    
    # TensorNEAT-specific helpers

    def build_neat_config(self, params: Dict[str, Any]) -> Any:
        """Build NEAT algorithm configuration using centralized builder.

        This method replaces duplicated parameter extraction code by delegating
        to TensorNEATConfigGenerator.

        Args:
            params: Parameter dictionary from web UI, YAML config, or grid search

        Returns:
            Configured TensorNEAT NEAT algorithm instance
        """
        from emr_hyperneat._compat.algorithms.tensorneat_config_generator import TensorNEATConfigGenerator
        import jax
        from tensorneat.common import ACT, AGG

        # Create schema-based builder and delegate
        builder = TensorNEATConfigGenerator.from_dict(params)
        return builder.build_neat_config(jax, ACT, AGG)

    def build_hyperneat_cppn_config(self, params: Dict[str, Any]):
        """Build CPPN NEAT configuration for HyperNEAT using centralized builder.

        This method replaces duplicated CPPN configuration code by delegating
        to TensorNEATConfigGenerator. Substrate creation
        is left to implementations using SubstrateConfigManager.

        Args:
            params: Parameter dictionary

        Returns:
            Configured CPPN NEAT algorithm
        """
        from emr_hyperneat._compat.algorithms.tensorneat_config_generator import TensorNEATConfigGenerator
        import jax
        from tensorneat.common import ACT, AGG

        # Ensure 'algorithm' key is set for proper HyperNEAT detection
        # Without this, from_dict() defaults to 'neat' and creates NEATConfig instead of HyperNEATConfig
        params_with_algo = params.copy()
        if 'algorithm' not in params_with_algo:
            params_with_algo['algorithm'] = 'hyperneat'

        # Create schema-based builder and delegate
        builder = TensorNEATConfigGenerator.from_dict(params_with_algo)
        return builder.build_hyperneat_cppn_config(jax, ACT, AGG)

    def create_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create TensorNEAT-compatible configuration.

        Args:
            params: Parameter dictionary

        Returns:
            TensorNEAT-compatible config dict
        """
        # Handle precision configuration
        if 'precision' in params:
            from emr_hyperneat._compat.utils.precision_handler import PrecisionHandler
            
            # Get hardware info
            hardware_config = params.get('hardware_config', {})
            has_gpu = hardware_config.get('device', 'auto') == 'gpu' or self._detect_gpu()
            problem_type = params.get('problem_type', 'classification')
            population_size = params.get('population_size', 150)
            
            # Configure precision
            actual_precision, requested_precision, precision_changed = PrecisionHandler.configure_precision(
                'tensorneat', 
                params['precision'], 
                has_gpu, 
                problem_type, 
                population_size
            )
            
            # Log warning if precision changed
            if precision_changed:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(PrecisionHandler.format_precision_warning(
                    'tensorneat', requested_precision, actual_precision
                ))
            
            # Update internal precision tracking
            self._precision = actual_precision
            # Set dtype based on the actual precision
            self._dtype = self._jnp.float32 if actual_precision == 'float32' else self._jnp.float64
        
        # Map framework parameters to TensorNEAT parameters
        config = {
            'population_size': params.get('population_size', 150),
            'max_generations': params.get('max_generations', 300),
            'fitness_threshold': self.get_fitness_threshold(
                params.get('normalized_threshold', 0.98)
            ),
            'seed': params.get('seed', 42),
            'verbose': params.get('verbose', False)
        }
        
        # Species parameters
        if 'species_size' in params:
            config['species_size'] = params['species_size']
        if 'compatibility_threshold' in params:
            # TensorNEAT uses different parameter names
            config['species_threshold'] = params['compatibility_threshold']
        
        # Mutation rates
        mutation_params = [
            'conn_add_prob', 'conn_delete_prob',
            'node_add_prob', 'node_delete_prob',
            'weight_mutate_rate', 'weight_mutate_power',
            'bias_mutate_rate', 'bias_mutate_power'
        ]
        
        for param in mutation_params:
            if param in params:
                config[param] = params[param]
        
        # Handle activation functions
        if 'activation_default' in params:
            config['activation_default'] = params['activation_default']
        
        # Copy other parameters
        for key, value in params.items():
            if key not in config and not key.startswith('_'):
                config[key] = value
        
        return config
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available using JAX.
        
        Returns:
            True if GPU is available
        """
        try:
            self._ensure_jax_initialized()
            devices = self._jax.devices()
            return any(d.platform == 'gpu' for d in devices)
        except:
            return False
    
    def set_precision(self, precision: str):
        """Set precision for TensorNEAT computations.
        
        Args:
            precision: 'float32' or 'float64'
            
        Note:
            This updates the internal dtype but does not change JAX's global
            configuration. For full precision control, set JAX_ENABLE_X64
            environment variable before importing TensorNEAT.
        """
        if precision not in ['float32', 'float64']:
            raise ValueError(f"Invalid precision: {precision}")
        
        super().set_precision(precision)
        # Update JAX dtype based on precision
        if self._jax_initialized:
            self._dtype = self._jnp.float32 if precision == 'float32' else self._jnp.float64
    
    # Implementation of new abstract methods
    
    def validate_network_connectivity(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a network has valid paths from inputs to outputs.
        
        Args:
            network_info: Dictionary containing network structure information
            
        Returns:
            Dictionary with validation results
        """
        logger = logging.getLogger(__name__)
        
        # Extract network structure
        connections = network_info.get('connections', [])
        nodes = network_info.get('nodes', {})
        
        # Build adjacency list for path finding
        graph = {}
        reverse_graph = {}  # For analyzing unreachable nodes
        connection_details = []  # Store connection details for reporting
        
        for conn in connections:
            if len(conn) >= 2:
                from_node = conn[0]
                to_node = conn[1]
                weight = conn[2] if len(conn) > 2 else 1.0
                enabled = conn[3] if len(conn) > 3 else True
                
                connection_details.append({
                    'from': from_node,
                    'to': to_node,
                    'weight': weight,
                    'enabled': enabled
                })
                
                if enabled:  # Only add enabled connections to graph
                    if from_node not in graph:
                        graph[from_node] = []
                    graph[from_node].append(to_node)
                    
                    if to_node not in reverse_graph:
                        reverse_graph[to_node] = []
                    reverse_graph[to_node].append(from_node)
        
        # Find input and output node IDs
        input_nodes = []
        output_nodes = []
        hidden_nodes = []
        
        for node_id, node_info in nodes.items():
            node_type = node_info.get('type', '')
            if node_type == 'input':
                input_nodes.append(node_id)
            elif node_type == 'output':
                output_nodes.append(node_id)
            elif node_type == 'hidden':
                hidden_nodes.append(node_id)
        
        # Check connectivity from each input to any output
        connected_inputs = 0
        disconnected_inputs = []
        input_paths = {}  # Store shortest path lengths
        
        for input_node in input_nodes:
            # BFS to check if this input reaches any output
            visited = set()
            queue = [(input_node, 0)]  # (node, distance)
            reaches_output = False
            min_path_length = float('inf')
            
            while queue:
                current, dist = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Check if we reached an output
                if current in output_nodes:
                    reaches_output = True
                    min_path_length = min(min_path_length, dist)
                
                # Add neighbors to queue
                if current in graph:
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))
            
            if reaches_output:
                connected_inputs += 1
                input_paths[input_node] = min_path_length
            else:
                disconnected_inputs.append(input_node)
        
        # Analyze unreachable nodes
        unreachable_nodes = []
        for node_id in nodes:
            if node_id not in input_nodes and node_id not in output_nodes:
                # Check if this node can be reached from any input
                reachable_from_input = False
                for input_node in input_nodes:
                    visited = set()
                    queue = [input_node]
                    while queue:
                        current = queue.pop(0)
                        if current in visited:
                            continue
                        visited.add(current)
                        if current == node_id:
                            reachable_from_input = True
                            break
                        if current in graph:
                            queue.extend(graph[current])
                    if reachable_from_input:
                        break
                
                if not reachable_from_input:
                    unreachable_nodes.append(node_id)
        
        # Calculate validation metrics
        is_valid = connected_inputs > 0  # At least one input must reach output
        connectivity_score = connected_inputs / len(input_nodes) if input_nodes else 0
        
        # Count enabled connections
        enabled_connections = sum(1 for c in connection_details if c['enabled'])
        
        # Create detailed topology report
        topology_report = {
            'node_counts': {
                'inputs': len(input_nodes),
                'outputs': len(output_nodes),
                'hidden': len(hidden_nodes),
                'total': len(nodes)
            },
            'connection_stats': {
                'total': len(connections),
                'enabled': enabled_connections,
                'disabled': len(connections) - enabled_connections
            },
            'connectivity': {
                'connected_inputs': connected_inputs,
                'disconnected_inputs': disconnected_inputs,
                'unreachable_nodes': unreachable_nodes,
                'input_path_lengths': input_paths
            }
        }
        
        validation_result = {
            'is_valid': is_valid,
            'connected_inputs': connected_inputs,
            'total_inputs': len(input_nodes),
            'disconnected_inputs': disconnected_inputs,
            'connectivity_score': connectivity_score,
            'num_connections': len(connections),
            'has_outputs': len(output_nodes) > 0,
            'topology_report': topology_report
        }
        
        if not is_valid:
            # Detailed report of the failed validation
            logger.warning(
                f"\n=== TensorNEAT Network Validation Failed ===\n"
                f"Connectivity: {connected_inputs}/{len(input_nodes)} inputs connected to outputs\n"
                f"Connections: {enabled_connections} enabled, {len(connections) - enabled_connections} disabled\n"
                f"Unreachable nodes: {len(unreachable_nodes)}\n"
                f"Disconnected inputs: {disconnected_inputs}\n"
                f"==========================================\n"
            )
            
            # Provide actionable suggestions
            if len(connections) == 0:
                logger.warning("  -> No connections in network! Check init_hidden_layers configuration.")
            elif enabled_connections == 0:
                logger.warning("  -> All connections are disabled! Check mutation settings.")
            elif len(unreachable_nodes) > len(hidden_nodes) * 0.5:
                logger.warning(f"  -> {len(unreachable_nodes)} nodes are unreachable. Consider increasing connection density.")
            
            # Show ASCII visualization for small networks
            if len(nodes) <= 10:
                logger.warning("\nNetwork visualization:")
                ascii_viz = self.visualize_network_ascii(network_info)
                for line in ascii_viz.split('\n'):
                    logger.warning(line)
        
        return validation_result
    
    def visualize_network_ascii(self, network_info: Dict[str, Any], max_nodes: int = 10) -> str:
        """Create ASCII visualization of a small network for debugging.
        
        Args:
            network_info: Dictionary containing network structure
            max_nodes: Maximum nodes to visualize (default 10)
            
        Returns:
            ASCII string representation of the network
        """
        nodes = network_info.get('nodes', {})
        connections = network_info.get('connections', [])
        
        if len(nodes) > max_nodes:
            return f"Network too large to visualize ({len(nodes)} nodes > {max_nodes} max)"
        
        # Categorize nodes
        input_nodes = []
        output_nodes = []
        hidden_nodes = []
        
        for node_id, node_info in nodes.items():
            node_type = node_info.get('type', '')
            if node_type == 'input':
                input_nodes.append(node_id)
            elif node_type == 'output':
                output_nodes.append(node_id)
            else:
                hidden_nodes.append(node_id)
        
        # Build connection map
        conn_map = {}
        for conn in connections:
            if len(conn) >= 2:
                from_node = conn[0]
                to_node = conn[1]
                enabled = conn[3] if len(conn) > 3 else True
                if enabled:
                    if from_node not in conn_map:
                        conn_map[from_node] = []
                    conn_map[from_node].append(to_node)
        
        # Create ASCII visualization
        lines = []
        lines.append("=== Network Topology ===")
        lines.append("")
        
        # Input layer
        lines.append("INPUTS:")
        for inp in sorted(input_nodes):
            connections_str = ""
            if inp in conn_map:
                connections_str = f" -> {conn_map[inp]}"
            lines.append(f"  [{inp}]{connections_str}")
        
        # Hidden layer
        if hidden_nodes:
            lines.append("")
            lines.append("HIDDEN:")
            for hid in sorted(hidden_nodes):
                connections_str = ""
                if hid in conn_map:
                    connections_str = f" -> {conn_map[hid]}"
                lines.append(f"  [{hid}]{connections_str}")
        
        # Output layer
        lines.append("")
        lines.append("OUTPUTS:")
        for out in sorted(output_nodes):
            # Find which nodes connect to this output
            incoming = []
            for from_node, to_nodes in conn_map.items():
                if out in to_nodes:
                    incoming.append(from_node)
            incoming_str = f" <- {incoming}" if incoming else " (disconnected)"
            lines.append(f"  [{out}]{incoming_str}")
        
        # Summary
        lines.append("")
        lines.append(f"Total connections: {len([c for c in connections if (len(c) < 4 or c[3])])}")
        lines.append("======================")
        
        return "\n".join(lines)
    
    def extract_network_info(self, result: Any) -> 'NetworkInfo':
        """Extract standardized network information from TensorNEAT result.
        
        Args:
            result: Result object from TensorNEAT experiment
            
        Returns:
            Standardized NetworkInfo object
        """
        from .neat_adapter import NetworkInfo
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Debug: Log what we're receiving
        logger.debug(f"extract_network_info called with result type: {type(result)}")
        if hasattr(result, 'network_info'):
            logger.debug(f"result.network_info type: {type(result.network_info)}")
            logger.debug(f"result.network_info content preview: {str(result.network_info)[:200]}...")
        else:
            logger.debug("result has no network_info attribute")
        
        # Check if result already has processed NetworkInfo object
        if hasattr(result, 'network_info') and result.network_info is not None:
            if isinstance(result.network_info, NetworkInfo):
                logger.debug("extract_network_info: Using already-processed NetworkInfo object")
                return result.network_info
        
        # Extract from network_info dictionary in the result - only if not already processed
        network_dict = result.network_info if (hasattr(result, 'network_info') and 
                                              not isinstance(result.network_info, NetworkInfo)) else {}
        
        # Debug: Log what we extracted
        logger.debug(f"Extracted network_dict type: {type(network_dict)}")
        if isinstance(network_dict, dict):
            logger.debug(f"Network dict keys: {list(network_dict.keys())}")
            logger.debug(f"Connections count: {len(network_dict.get('connections', []))}")
        else:
            logger.debug(f"Network dict is not a dict: {network_dict}")
        
        # Only validate if we have dictionary data to process
        if isinstance(network_dict, dict):
            # Skip validation if this looks like an empty/post-processed dict from ExperimentResult
            # (which indicates the network was already validated during evolution)
            if not network_dict or (len(network_dict.get('connections', [])) == 0 and 
                                   network_dict.get('num_connections', 0) == 0):
                logger.debug("Skipping validation for empty network dict (likely from successful experiment)")
                return NetworkInfo(
                    num_nodes=network_dict.get('num_nodes', 0),
                    num_connections=network_dict.get('num_connections', 0),
                    num_enabled_connections=network_dict.get('num_connections', 0),
                    num_inputs=network_dict.get('num_inputs', 0),
                    num_outputs=network_dict.get('num_outputs', 0),
                    num_hidden=network_dict.get('num_hidden', 0),
                    connections=network_dict.get('connections', []),
                    nodes=network_dict.get('nodes', {})
                )
            
            # Validate network connectivity only for raw data
            validation = self.validate_network_connectivity(network_dict)

            # Log if network is disconnected (debug only - may occur during intermediate extraction)
            if not validation['is_valid']:
                logger.debug(
                    f"Extracted TensorNEAT network with partial connectivity: "
                    f"{validation['connected_inputs']}/{validation['total_inputs']} inputs connected"
                )
            
            return NetworkInfo(
                num_nodes=network_dict.get('num_nodes', 0),
                num_connections=network_dict.get('num_connections', 0),
                num_enabled_connections=network_dict.get('num_connections', 0),  # May need to check enabled field
                num_inputs=network_dict.get('num_inputs', 0),
                num_outputs=network_dict.get('num_outputs', 0),
                num_hidden=network_dict.get('num_hidden', 0),
                connections=network_dict.get('connections', []),
                nodes=network_dict.get('nodes', {})
            )
        else:
            # Unknown format - return empty NetworkInfo
            logger.warning(f"extract_network_info: Unknown network_info format: {type(network_dict)}")
            return NetworkInfo(
                num_nodes=0,
                num_connections=0,
                num_enabled_connections=0,
                num_inputs=0,
                num_outputs=0,
                num_hidden=0,
                connections=[],
                nodes={}
            )
    
    
    
    
    
    
