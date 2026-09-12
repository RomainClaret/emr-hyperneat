"""
TensorNEAT Implementation Adapter.

This adapter handles all TensorNEAT-specific conversions and requirements.
"""

from typing import Any, Dict, List, Union, Optional
import numpy as np
import os
import logging

from .neat_adapter import NEATAdapter


class TensorNEATAdapter(NEATAdapter):
    """Adapter for TensorNEAT implementation.
    
    TensorNEAT characteristics:
    - Does NOT require bias input (bias is built into nodes)
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
    
    # Input Preparation Methods
    
    def prepare_input(self, raw_input: np.ndarray) -> Any:
        """Prepare input for TensorNEAT (no bias needed).
        
        Args:
            raw_input: Raw input array from problem
            
        Returns:
            JAX array without bias
        """
        self._ensure_jax_initialized()
        # TensorNEAT has bias built into nodes, so no need to add
        return self._jnp.array(raw_input, dtype=self._dtype)
    
    def prepare_batch(self, inputs: List[np.ndarray]) -> Any:
        """Prepare batch of inputs for TensorNEAT.
        
        Args:
            inputs: List of raw input arrays
            
        Returns:
            JAX array batch
        """
        self._ensure_jax_initialized()
        # Stack inputs into a JAX array
        batch = np.stack(inputs)
        return self._jnp.array(batch, dtype=self._dtype)
    
    def needs_bias(self) -> bool:
        """TensorNEAT does NOT require bias in inputs.
        
        Returns:
            False
        """
        return False
    
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
            
            # Check for NaN in outputs with enhanced diagnostics
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
    
    def activate_batch(self, network: Any, prepared_batch: Any) -> np.ndarray:
        """Activate TensorNEAT network on a batch.
        
        Args:
            network: TensorNEAT network function
            prepared_batch: JAX array batch from prepare_batch()
            
        Returns:
            Array of outputs
        """
        self._ensure_jax_initialized()
        # Use JAX vmap for efficient vectorized execution
        network_vmap = self._jax.vmap(network)
        outputs = network_vmap(prepared_batch)
        
        # Convert to numpy
        result = np.array(outputs)
        # Keep the same dtype as configured
        return result
    
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
    
    # Metric Conversion Methods
    
    def convert_metrics(self, raw_metrics: Dict[str, float],
                       fitness_range: tuple[float, float] = (-1.0, 0.0)) -> Dict[str, float]:
        """Convert TensorNEAT metrics to normalized values.
        
        Args:
            raw_metrics: Dictionary of raw metrics
            fitness_range: (min, max) fitness range from the problem
            
        Returns:
            Dictionary of normalized metrics
        """
        normalized = raw_metrics.copy()
        
        # Normalize fitness if present (from negative scale to [0, 1])
        if 'fitness' in normalized:
            normalized['fitness'] = self.normalize_fitness(normalized['fitness'], fitness_range)
        
        # Handle MSE - should be positive
        if 'mse' in normalized and normalized['mse'] < 0:
            normalized['mse'] = -normalized['mse']
        
        return normalized
    
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

        # FIX: Ensure 'algorithm' key is set for proper HyperNEAT detection
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
    
    def is_jit_enabled(self) -> bool:
        """Check if JAX JIT compilation is enabled.
        
        Returns:
            True if JIT is enabled
        """
        return os.environ.get('JAX_DISABLE_JIT', '').lower() not in ('1', 'true', 'yes')
    
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
    
    def prepare_for_jit(self, network: Any) -> Any:
        """Prepare network for JIT compilation if needed.
        
        Args:
            network: TensorNEAT network function
            
        Returns:
            JIT-compiled network if JIT is enabled
        """
        if self.is_jit_enabled():
            self._ensure_jax_initialized()
            return self._jax.jit(network)
        return network
    
    def supports_precision_control(self) -> bool:
        """Check if this adapter supports precision control.
        
        TensorNEAT uses JAX backend and respects JAX's precision settings.
        Precision can be controlled via the JAX_ENABLE_X64 environment variable.
        
        Returns:
            True (TensorNEAT supports both float32 and float64)
        """
        return True
    
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
    
    def get_implementation_type(self) -> str:
        """Get the implementation type identifier.
        
        Returns:
            'tensorneat'
        """
        return 'tensorneat'
    
    def normalize_fitness_with_context(self, raw_fitness: float, problem: str = None) -> float:
        """Convert TensorNEAT fitness to normalized [0, 1] range with problem context.
        
        Args:
            raw_fitness: Fitness value from TensorNEAT (1.0 - MSE)
            problem: Optional problem name for context-aware normalization
            
        Returns:
            Normalized fitness in [0, 1] range
        """
        # Handle NaN/Inf - these represent network evaluation failures
        if np.isnan(raw_fitness):
            # NaN typically means disconnected network or numerical instability
            # Return very low fitness to clearly indicate failure
            logger = logging.getLogger(__name__)
            logger.error(
                "\n=== TensorNEAT NaN Fitness Detected ===\n"
                "Network evaluation failed - common causes:\n"
                "1. Disconnected network (no input→output paths)\n"
                "2. Numerical instability in computations\n"
                "3. Invalid network configuration\n"
                "\nDiagnostic steps:\n"
                "• Check init_hidden_layers configuration (use [] for direct connections)\n"
                "• Verify network connectivity with validate_network_connectivity()\n"
                "• Note: Connection mutation has only 2-7% success rate\n"
                "• Consider increasing population size or initial connections\n"
                "======================================\n"
            )
            return 0.0  # Complete failure
        
        if np.isinf(raw_fitness):
            # Infinity means extreme error
            logger = logging.getLogger(__name__)
            logger.error(
                f"\n=== TensorNEAT Infinite Fitness Detected ===\n"
                f"Raw fitness value: {raw_fitness}\n"
                f"This indicates extreme error in network output.\n"
                f"\nPossible causes:\n"
                f"• Exploding gradients or weights\n"
                f"• Division by zero in custom fitness functions\n"
                f"• Numerical overflow in computations\n"
                f"==========================================\n"
            )
            return 0.0  # Complete failure
        
        # TensorNEAT already outputs fitness in [0, 1] range
        # Just ensure bounds
        return min(1.0, max(0.0, raw_fitness))
    
    def validate_network_connectivity(self, network_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a network has valid paths from inputs to outputs.
        
        Args:
            network_info: Dictionary containing network structure information
            
        Returns:
            Dictionary with validation results
        """
        logger = logging.getLogger(__name__)
        
        # Extract network structure
        num_inputs = network_info.get('num_inputs', 0)
        num_outputs = network_info.get('num_outputs', 0)
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
            # Enhanced error reporting
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
                logger.warning("  → No connections in network! Check init_hidden_layers configuration.")
            elif enabled_connections == 0:
                logger.warning("  → All connections are disabled! Check mutation settings.")
            elif len(unreachable_nodes) > len(hidden_nodes) * 0.5:
                logger.warning(f"  → {len(unreachable_nodes)} nodes are unreachable. Consider increasing connection density.")
            
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
                connections_str = f" → {conn_map[inp]}"
            lines.append(f"  [{inp}]{connections_str}")
        
        # Hidden layer
        if hidden_nodes:
            lines.append("")
            lines.append("HIDDEN:")
            for hid in sorted(hidden_nodes):
                connections_str = ""
                if hid in conn_map:
                    connections_str = f" → {conn_map[hid]}"
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
            incoming_str = f" ← {incoming}" if incoming else " (disconnected)"
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
        import numpy as np
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
    
    def extract_network_info_from_genome(self, nodes: Any, connections: Any, 
                                        num_inputs: int, num_outputs: int) -> Dict[str, Any]:
        """Extract network information from TensorNEAT genome arrays.
        
        TensorNEAT pre-allocates arrays to max_nodes and max_conns size,
        with NaN values marking unused entries.
        
        Args:
            nodes: Node array from TensorNEAT genome
            connections: Connection array from TensorNEAT genome
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
            
        Returns:
            Dictionary with network information
        """
        import numpy as np
        
        # Count valid nodes (non-NaN in first column)
        nodes_array = np.array(nodes)
        valid_node_mask = ~np.isnan(nodes_array[:, 0]) if nodes_array.ndim > 1 else np.ones(len(nodes_array), dtype=bool)
        valid_nodes = int(np.sum(valid_node_mask))
        
        # Count valid connections (non-NaN in first column)
        conns_array = np.array(connections)
        valid_conn_mask = ~np.isnan(conns_array[:, 0]) if conns_array.ndim > 1 else np.ones(len(conns_array), dtype=bool)
        valid_connections = int(np.sum(valid_conn_mask))
        
        # Extract node information
        node_dict = {}
        for i, is_valid in enumerate(valid_node_mask):
            if is_valid:
                node_id = int(nodes_array[i, 0])
                # Determine node type based on ID
                if node_id < num_inputs:
                    node_type = 'input'
                elif node_id < num_inputs + num_outputs:
                    node_type = 'output'
                else:
                    node_type = 'hidden'
                
                node_dict[node_id] = {
                    'type': node_type,
                    'bias': float(nodes_array[i, 1]) if nodes_array.shape[1] > 1 else 0.0,
                    'activation': 'tanh',  # Default, would need to decode from array
                    'aggregation': 'sum'   # Default
                }
        
        # Extract connection information
        connection_list = []
        for i, is_valid in enumerate(valid_conn_mask):
            if is_valid and conns_array.shape[1] >= 3:
                connection_list.append((
                    int(conns_array[i, 0]),    # from_node
                    int(conns_array[i, 1]),    # to_node
                    float(conns_array[i, 2]),  # weight
                    True                        # enabled (TensorNEAT doesn't store this)
                ))
        
        num_hidden = valid_nodes - num_inputs - num_outputs
        
        return {
            'num_nodes': valid_nodes,
            'num_connections': valid_connections,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'num_hidden': max(0, num_hidden),
            'connections': connection_list,
            'nodes': node_dict
        }
    
    def extract_evolution_metrics(self, result: Any, generation: int = None) -> 'EvolutionMetrics':
        """Extract standardized evolution metrics from TensorNEAT result.
        
        Args:
            result: Result object from TensorNEAT experiment
            generation: Optional generation number if not in result
            
        Returns:
            Standardized EvolutionMetrics object
        """
        from .neat_adapter import EvolutionMetrics
        
        # Get the last metrics from history if available
        if hasattr(result, 'metrics_history') and result.metrics_history:
            last_metrics = result.metrics_history[-1]
            if hasattr(last_metrics, '__dict__'):
                # It's an object with attributes
                return EvolutionMetrics(
                    generation=generation or last_metrics.generation,
                    best_fitness=last_metrics.best_fitness,
                    mean_fitness=last_metrics.mean_fitness,
                    min_fitness=last_metrics.min_fitness,
                    max_fitness=last_metrics.max_fitness,
                    std_fitness=last_metrics.std_fitness,
                    num_species=last_metrics.num_species,
                    species_sizes=last_metrics.species_sizes,
                    species_fitness=last_metrics.species_fitness,
                    evaluations=last_metrics.evaluations
                )
            elif isinstance(last_metrics, dict):
                # It's already a dictionary
                return EvolutionMetrics(
                    generation=generation or last_metrics.get('generation', 0),
                    best_fitness=last_metrics.get('best_fitness', 0.0),
                    mean_fitness=last_metrics.get('mean_fitness', 0.0),
                    min_fitness=last_metrics.get('min_fitness', 0.0),
                    max_fitness=last_metrics.get('max_fitness', 0.0),
                    std_fitness=last_metrics.get('std_fitness', 0.0),
                    num_species=last_metrics.get('num_species', 0),
                    species_sizes=last_metrics.get('species_sizes', []),
                    species_fitness=last_metrics.get('species_fitness', []),
                    evaluations=last_metrics.get('evaluations', 0)
                )
        
        # Fallback to basic info from result
        return EvolutionMetrics(
            generation=generation or getattr(result, 'generations', 0),
            best_fitness=getattr(result, 'final_fitness', 0.0),
            mean_fitness=0.0,
            min_fitness=0.0,
            max_fitness=getattr(result, 'final_fitness', 0.0),
            std_fitness=0.0,
            num_species=0,
            species_sizes=[],
            species_fitness=[],
            evaluations=0
        )
    
    def calculate_performance_overhead(self, result: Any) -> Dict[str, float]:
        """Calculate TensorNEAT-specific performance overhead.
        
        TensorNEAT has JIT compilation overhead from JAX.
        
        Args:
            result: Result object from TensorNEAT experiment
            
        Returns:
            Dictionary with JIT overhead metrics
        """
        overhead = {}
        
        # Extract JIT overhead if available in metrics
        if hasattr(result, 'metrics_history') and result.metrics_history:
            # Check if the metrics have JIT timing information
            for metric in result.metrics_history:
                custom_metrics = None
                if hasattr(metric, 'custom_metrics'):
                    custom_metrics = metric.custom_metrics
                elif isinstance(metric, dict):
                    custom_metrics = metric.get('custom_metrics', {})
                
                if custom_metrics and 'timing' in custom_metrics:
                    timing = custom_metrics['timing']
                    if 'jit_transform' in timing:
                        overhead['jit_transform_time'] = timing['jit_transform']
                    if 'jit_eval' in timing:
                        overhead['jit_eval_time'] = timing['jit_eval']
                    if 'jit_to_jax' in timing:
                        overhead['jit_to_jax_time'] = timing['jit_to_jax']
                    if 'jit_to_python' in timing:
                        overhead['jit_to_python_time'] = timing['jit_to_python']
                    break  # Use first available timing data
        
        # Calculate total JIT overhead
        if any(k.startswith('jit_') for k in overhead):
            total_jit = sum(v for k, v in overhead.items() if k.startswith('jit_'))
            overhead['total_jit_overhead'] = total_jit
        
        return overhead
    
    # Parameter Mapping Methods
    
    def map_to_implementation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map standard parameters to implementation-specific parameters.
        
        This is a wrapper for to_native_config() to match the expected interface.
        
        Args:
            params: Standard parameter dictionary
            
        Returns:
            Implementation-specific parameter dictionary
        """
        return self.to_native_config(params)
    
    def get_parameter_mapping(self) -> Dict[str, str]:
        """Get parameter name mapping from unified names to TensorNEAT-specific names.
        
        Returns:
            Dictionary mapping unified parameter names to TensorNEAT names
        """
        return {
            # Population parameters
            'population_size': 'pop_size',
            'elitism': 'genome_elitism',
            'species_elitism': 'species_elitism',
            'survival_threshold': 'survival_threshold',
            'min_species_size': 'min_species_size',
            
            # Mutation parameters
            'conn_add_prob': 'conn_add',
            'conn_delete_prob': 'conn_delete',
            'node_add_prob': 'node_add',
            'node_delete_prob': 'node_delete',
            
            # Weight parameters  
            'weight_mutate_rate': 'weight_mutate_rate',
            'weight_mutate_power': 'weight_mutate_power',
            'weight_replace_rate': 'weight_replace_rate',
            'weight_init_mean': 'weight_init_mean',
            'weight_init_stdev': 'weight_init_std',
            'weight_max_value': 'weight_upper_bound',
            'weight_min_value': 'weight_lower_bound',
            
            # Bias parameters
            'bias_mutate_rate': 'bias_mutate_rate',
            'bias_mutate_power': 'bias_mutate_power',
            'bias_replace_rate': 'bias_replace_rate',
            'bias_init_mean': 'bias_init_mean',
            'bias_init_stdev': 'bias_init_std',
            'bias_max_value': 'bias_upper_bound',
            'bias_min_value': 'bias_lower_bound',
            
            # Response parameters
            'response_mutate_rate': 'response_mutate_rate',
            'response_mutate_power': 'response_mutate_power',
            'response_replace_rate': 'response_replace_rate',
            'response_init_mean': 'response_init_mean',
            'response_init_stdev': 'response_init_std',
            
            # Species parameters
            'compatibility_threshold': 'compatibility_threshold',
            'max_stagnation': 'max_stagnation',
            'species_fitness_func': 'species_fitness_func',
            
            # Activation parameters
            'activation_default': 'activation_default',
            'activation_mutate_rate': 'activation_replace_rate',
            'activation_options': 'activation_options',
            
            # Aggregation parameters
            'aggregation_default': 'aggregation_default',
            'aggregation_mutate_rate': 'aggregation_replace_rate',
            'aggregation_options': 'aggregation_options',
            
            # Connection parameters
            'enabled_default': 'enabled_default',
            'enabled_mutate_rate': 'enabled_mutate_rate',
            
            # Network structure
            'feed_forward': 'feed_forward',
            'num_hidden': 'init_hidden_layers',
            'num_inputs': 'num_inputs',
            'num_outputs': 'num_outputs',
            
            # Evolution parameters
            'fitness_criterion': 'fitness_criterion',
            'fitness_threshold': 'fitness_threshold',
            'max_generations': 'max_generations',
            'reset_on_extinction': False  # TensorNEAT doesn't support this
        }
    
    def to_native_config(self, unified_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified configuration to TensorNEAT-specific format.
        
        Args:
            unified_config: Configuration using unified parameter names
            
        Returns:
            Configuration using TensorNEAT-specific parameter names
        """
        mapping = self.get_parameter_mapping()
        native_config = {}
        
        for unified_name, value in unified_config.items():
            if unified_name in mapping:
                mapped_value = mapping[unified_name]
                # Handle special cases where mapping is a value, not a name
                if isinstance(mapped_value, bool):
                    # Skip parameters TensorNEAT doesn't support
                    continue
                native_config[mapped_value] = value
            else:
                # Pass through unknown parameters
                native_config[unified_name] = value
        
        # Handle special cases
        # TensorNEAT expects activation options as list
        if 'activation_options' in native_config and isinstance(native_config['activation_options'], str):
            native_config['activation_options'] = native_config['activation_options'].split()
        
        return native_config
    
    def from_native_config(self, native_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TensorNEAT-specific configuration to unified format.
        
        Args:
            native_config: Configuration using TensorNEAT-specific names
            
        Returns:
            Configuration using unified parameter names
        """
        # Create reverse mapping
        mapping = self.get_parameter_mapping()
        reverse_mapping = {}
        for k, v in mapping.items():
            if isinstance(v, str):  # Only string values are actual mappings
                reverse_mapping[v] = k
        
        unified_config = {}
        for native_name, value in native_config.items():
            if native_name in reverse_mapping:
                unified_name = reverse_mapping[native_name]
                unified_config[unified_name] = value
            else:
                # Pass through unknown parameters
                unified_config[native_name] = value
        
        # Handle special cases
        # Ensure activation options is a list
        if 'activation_options' in unified_config and isinstance(unified_config['activation_options'], str):
            unified_config['activation_options'] = unified_config['activation_options'].split()
        
        return unified_config
    
    # Hardware and Parallelization Methods
    
    def get_hardware_requirements(self) -> Dict[str, Any]:
        """Get hardware requirements for TensorNEAT.
        
        TensorNEAT benefits from GPU acceleration and requires larger populations.
        
        Returns:
            Hardware requirements dictionary
        """
        return {
            'gpu_required': False,  # Can run on CPU
            'gpu_recommended': True,  # Much faster on GPU
            'min_memory_mb': 1024,
            'recommended_memory_mb': 4096,
            'min_population_size': 100,  # Below this, success rate drops
            'recommended_population_size': 300  # Optimal for most problems
        }
    
    def supports_multiprocessing(self) -> bool:
        """TensorNEAT with JIT doesn't work well with multiprocessing.
        
        JAX JIT compilation and multiprocessing can conflict.
        
        Returns:
            False
        """
        return False
    
    def get_optimal_parallelization(self, population_size: int, 
                                  num_cpus: int = None, 
                                  has_gpu: bool = False) -> Dict[str, Any]:
        """Get optimal parallelization configuration for TensorNEAT.
        
        Args:
            population_size: Size of the population
            num_cpus: Number of available CPUs (ignored)
            has_gpu: Whether GPU is available
            
        Returns:
            Parallelization configuration
        """
        # TensorNEAT uses JAX which handles its own parallelization
        # Multiprocessing conflicts with JAX JIT
        return {
            'use_multiprocessing': False,
            'n_processes': 1,
            'batch_size': population_size if has_gpu else min(population_size, 32),
            'notes': f"TensorNEAT using JAX parallelization (GPU: {has_gpu})"
        }
    
    # Problem Transformation Methods
    
    def transform_problem(self, problem: 'BaseProblem') -> 'BaseProblem':
        """Transform a problem to work with TensorNEAT.
        
        With the new implementation-agnostic design, problems no longer need
        to be configured for specific implementations. The adapter handles
        all necessary transformations through the data transformation methods.
        
        Args:
            problem: Original problem instance
            
        Returns:
            Same problem instance (no transformation needed)
        """
        # No transformation needed - adapters handle data transformation
        return problem
    
    # Capability Declaration Methods
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Declare TensorNEAT capabilities.
        
        Returns:
            Capability flags
        """
        return {
            'batch_evaluation': True,  # Vectorized evaluation
            'gpu_acceleration': True,  # JAX GPU support
            'distributed_evaluation': False,  # not implemented in this release
            'dynamic_population': True,  # Can change population size
            'checkpointing': True,  # Supports saving/loading
            'real_time_visualization': False,  # No built-in visualization
            'custom_mutations': True,  # Supports custom mutation operators
            'substrate_evolution': False  # Basic NEAT only (ES-HyperNEAT planned)
        }
    
    def get_limitations(self) -> List[str]:
        """Get TensorNEAT limitations.
        
        Returns:
            List of limitations
        """
        return [
            "Requires population >= 100 for reasonable success rates",
            "JIT compilation overhead on first generation",
            "Cannot use multiprocessing with JIT enabled",
            "May have memory issues with very large populations on GPU",
            "Limited to JAX-supported operations"
        ]
    
    # Evolution Dynamics Extraction
    
    def extract_species_count(self, result: Any) -> int:
        """Extract current species count from TensorNEAT result.
        
        Args:
            result: Result object from TensorNEAT
            
        Returns:
            Number of species
        """
        try:
            # TensorNEAT stores species count in metrics
            if hasattr(result, 'metrics_history') and result.metrics_history:
                # Get from last generation metrics
                last_metrics = result.metrics_history[-1]
                if hasattr(last_metrics, 'num_species'):
                    return last_metrics.num_species
                elif isinstance(last_metrics, dict):
                    return last_metrics.get('num_species', 0)
            # Fallback to network_info
            return result.network_info.get('num_species', 0)
        except Exception:
            return 0
    
    def extract_species_history(self, result: Any) -> List[int]:
        """Extract species count history from TensorNEAT result.
        
        Args:
            result: Result object from TensorNEAT
            
        Returns:
            List of species counts per generation
        """
        try:
            if hasattr(result, 'metrics_history') and result.metrics_history:
                history = []
                for metric in result.metrics_history:
                    if hasattr(metric, 'num_species'):
                        history.append(metric.num_species)
                    elif isinstance(metric, dict):
                        history.append(metric.get('num_species', 1))
                return history
            return []
        except Exception:
            return []
    
    def extract_species_populations(self, state: Any) -> Dict[int, int]:
        """Extract detailed species populations from TensorNEAT state.
        
        Args:
            state: TensorNEAT state object
            
        Returns:
            Dictionary mapping species_id to population size
        """
        try:
            import jax.numpy as jnp
            species_populations = {}
            
            # TensorNEAT stores species information in the state object
            if hasattr(state, 'species_set'):
                species_set = state.species_set
                
                # Extract species IDs and member counts
                if hasattr(species_set, 'species_ids') and hasattr(species_set, 'members'):
                    # species_ids contains the unique species identifiers
                    # members is typically a mapping or array of genome_id -> species_id
                    
                    # Convert JAX arrays to numpy for easier processing
                    if hasattr(species_set.species_ids, '__array__'):
                        species_ids = np.array(species_set.species_ids)
                    else:
                        species_ids = species_set.species_ids
                    
                    if hasattr(species_set.members, '__array__'):
                        members = np.array(species_set.members)
                    else:
                        members = species_set.members
                    
                    # Count members per species
                    for species_id in np.unique(species_ids):
                        if not np.isnan(species_id) and species_id >= 0:
                            # Count how many genomes belong to this species
                            count = np.sum(members == species_id)
                            if count > 0:
                                species_populations[int(species_id)] = int(count)
                
                # Alternative: Check if species_set has a different structure
                elif hasattr(species_set, 'species') and hasattr(species_set.species, '__len__'):
                    # Some TensorNEAT versions might store species as a list/dict
                    for idx, species in enumerate(species_set.species):
                        if hasattr(species, 'members'):
                            species_populations[idx] = len(species.members)
                        elif hasattr(species, '__len__'):
                            species_populations[idx] = len(species)
            
            # Fallback: try to extract from population if species_set not available
            elif hasattr(state, 'population'):
                # TensorNEAT might store species assignments in the population
                if hasattr(state.population, 'species_ids'):
                    species_ids = np.array(state.population.species_ids)
                    for species_id in np.unique(species_ids):
                        if not np.isnan(species_id) and species_id >= 0:
                            count = np.sum(species_ids == species_id)
                            if count > 0:
                                species_populations[int(species_id)] = int(count)
            
            return species_populations
        except Exception as e:
            logger.debug(f"Error extracting species populations from TensorNEAT: {e}")
            return {}
    
    def extract_species_fitnesses(self, state: Any) -> Dict[int, float]:
        """Extract best fitness per species from TensorNEAT state.
        
        Args:
            state: TensorNEAT state object
            
        Returns:
            Dictionary mapping species_id to best fitness in that species
        """
        try:
            import jax.numpy as jnp
            species_fitnesses = {}
            
            # TensorNEAT stores fitness and species information in the state
            if hasattr(state, 'population') and hasattr(state, 'species_set'):
                population = state.population
                species_set = state.species_set
                
                # Get fitness values and species assignments
                if hasattr(population, 'fitnesses') and hasattr(species_set, 'members'):
                    fitnesses = np.array(population.fitnesses)
                    members = np.array(species_set.members)  # genome_id -> species_id mapping
                    
                    # Find best fitness per species
                    for species_id in np.unique(members):
                        if not np.isnan(species_id) and species_id >= 0:
                            # Get indices of genomes in this species
                            species_mask = members == species_id
                            if np.any(species_mask):
                                # Get fitnesses for this species
                                species_fitness_values = fitnesses[species_mask]
                                # Filter out NaN values
                                valid_fitnesses = species_fitness_values[~np.isnan(species_fitness_values)]
                                if len(valid_fitnesses) > 0:
                                    species_fitnesses[int(species_id)] = float(np.max(valid_fitnesses))
            
            # Alternative approach if the structure is different
            elif hasattr(state, 'fitnesses') and hasattr(state, 'species_ids'):
                fitnesses = np.array(state.fitnesses)
                species_ids = np.array(state.species_ids)
                
                for species_id in np.unique(species_ids):
                    if not np.isnan(species_id) and species_id >= 0:
                        species_mask = species_ids == species_id
                        if np.any(species_mask):
                            species_fitness_values = fitnesses[species_mask]
                            valid_fitnesses = species_fitness_values[~np.isnan(species_fitness_values)]
                            if len(valid_fitnesses) > 0:
                                species_fitnesses[int(species_id)] = float(np.max(valid_fitnesses))
            
            return species_fitnesses
        except Exception as e:
            logger.debug(f"Error extracting species fitnesses from TensorNEAT: {e}")
            return {}
    
    # Success Criteria Methods
    
    def get_success_criteria(self, problem: str) -> Dict[str, Any]:
        """Get TensorNEAT-specific success criteria for a problem.
        
        TensorNEAT uses positive fitness (1.0 - MSE).
        
        Args:
            problem: Problem name
            
        Returns:
            Success criteria dictionary
        """
        if problem == 'xor':
            return {
                'fitness_threshold': 0.95,  # 95% accuracy
                'min_generations': 10,  # Give JIT time to compile
                'max_stagnation': 20
            }
        elif problem == 'parity':
            return {
                'fitness_threshold': 0.95,
                'min_generations': 10,
                'max_stagnation': 25
            }
        elif problem == 'sine_approximation':
            return {
                'fitness_threshold': 0.98,  # Stricter for regression
                'min_generations': 10,
                'max_stagnation': 30
            }
        elif problem == 'xor_generalization':
            return {
                'fitness_threshold': 0.95,
                'min_generations': 10,
                'max_stagnation': 25
            }
        else:
            # Default criteria
            return {
                'fitness_threshold': 0.95,
                'min_generations': 10,
                'max_stagnation': 25
            }
    
    def is_success(self, fitness: float, problem: str) -> bool:
        """Check if a fitness value represents success for a problem.
        
        Args:
            fitness: Raw fitness value from TensorNEAT (1.0 - MSE)
            problem: Problem name
            
        Returns:
            True if fitness meets success criteria
        """
        criteria = self.get_success_criteria(problem)
        threshold = criteria['fitness_threshold']
        
        # For TensorNEAT, fitness is positive, so higher is better
        return fitness >= threshold
    
    def get_problem_compatibility(self, problem: str) -> float:
        """Get compatibility score for a problem.
        
        Args:
            problem: Problem name
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # TensorNEAT is optimized for batch processing and GPU acceleration
        compatibility_scores = {
            'xor': 0.8,  # Good for XOR but overkill
            'parity': 0.9,  # Good for parity with batching
            'sine_approximation': 0.9,  # Good for regression with JAX
            'xor_generalization': 0.9,  # Good for generalization
            'mnist': 1.0,  # Excellent for image tasks with GPU
            'hyperneat': 0.0,  # Not a HyperNEAT implementation yet
        }
        
        # Default compatibility for unknown problems
        return compatibility_scores.get(problem, 0.8)
    
    def get_default_config(self, problem: str, population_size: int = None) -> Dict[str, Any]:
        """Get default TensorNEAT configuration for a problem.
        
        Args:
            problem: Problem name
            population_size: Optional population size
            
        Returns:
            Default configuration dictionary in TensorNEAT format
        """
        if population_size is None:
            population_size = 300  # TensorNEAT needs larger populations
            
        base_config = {
            'pop_size': population_size,
            'fitness_criterion': 'max',
            'genome_elitism': 2,
            'species_elitism': 2,
            'survival_threshold': 0.2,
            'min_species_size': 2,
            'max_stagnation': 20,
            'species_fitness_func': 'max',
            
            # Structure
            'feed_forward': True,
            'init_hidden_layers': [],  # Empty list = direct input→output connections
            'initial_connection': 'full',  # CRITICAL: Ensure networks start with connections!
            
            # Mutation rates (TensorNEAT naming)
            'conn_add': 0.5,
            'conn_delete': 0.5,
            'node_add': 0.2,
            'node_delete': 0.2,
            
            # Weight parameters
            'weight_mutate_rate': 0.8,
            'weight_mutate_power': 0.5,
            'weight_replace_rate': 0.1,
            'weight_init_mean': 0.0,
            'weight_init_std': 1.0,
            'weight_upper_bound': 30,
            'weight_lower_bound': -30,
            
            # Bias parameters
            'bias_mutate_rate': 0.7,
            'bias_mutate_power': 0.5,
            'bias_replace_rate': 0.1,
            'bias_init_mean': 0.0,
            'bias_init_std': 1.0,
            'bias_upper_bound': 30,
            'bias_lower_bound': -30,
            
            # Response parameters
            'response_mutate_rate': 0.0,
            'response_mutate_power': 0.0,
            'response_replace_rate': 0.0,
            'response_init_mean': 1.0,
            'response_init_std': 0.0,
            
            # Activation
            'activation_default': 'sigmoid',
            'activation_replace_rate': 0.0,
            'activation_options': ['sigmoid'],
            
            # Aggregation
            'aggregation_default': 'sum',
            'aggregation_replace_rate': 0.0,
            'aggregation_options': ['sum'],
            
            # Connection
            'enabled_default': True,
            'enabled_mutate_rate': 0.01,
            
            # Species
            'compatibility_threshold': 3.0,
            
            # TensorNEAT specific
            'algorithm': 'neat',
            'use_jit': True,
            'use_smart_mutations': False,  # Default to False for adaptive mode (fast first, retry if needed)
            'seed': 42,
            'verbose': False
        }
        
        # Problem-specific overrides
        if problem == 'xor':
            base_config['fitness_threshold'] = 0.95
            base_config['max_generations'] = 300
        elif problem == 'parity':
            base_config['fitness_threshold'] = 0.95
            base_config['max_generations'] = 500
            base_config['max_stagnation'] = 25
        elif problem == 'sine_approximation':
            base_config['fitness_threshold'] = 0.98
            base_config['max_generations'] = 500
            base_config['activation_options'] = ['sigmoid', 'tanh', 'relu']
        elif problem == 'xor_generalization':
            base_config['fitness_threshold'] = 0.95
            base_config['max_generations'] = 400
            
        return base_config