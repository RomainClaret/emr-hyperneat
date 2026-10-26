"""Configuration manager for unified hyperparameter management."""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..config.parameter_translator import ParameterTranslator

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""
    pass


class ConfigParseError(ConfigError):
    """Raised when a configuration file cannot be parsed."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when a configuration fails validation."""
    pass


class ConfigManager:
    """Manages hierarchical configuration loading for experiment framework."""
    
    def __init__(self, config_base_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_base_path: Base path for configuration files. 
                            Defaults to experiment_framework/configs
        """
        if config_base_path is None:
            # Default to configs directory relative to this file
            config_base_path = Path(__file__).parent.parent / 'configs'
        
        self.config_base_path = Path(config_base_path)
        
        # Configuration hierarchy paths
        self.base_config_path = self.config_base_path / 'base'
        self.impl_config_path = self.config_base_path / 'implementations'

        # Implementation variant mapping to config directories
        # Maps implementation runtime names to their config directory names
        # Includes both bare names (from Web UI) and suffixed names (from catalog)
        self.impl_config_map = {
            'tensorneat-compiled': 'tensorneat',
            'tensorneat-jit': 'tensorneat',
            'tensorneat-jax-steps': 'tensorneat',              # Web UI bare name
            'tensorneat-optimized': 'tensorneat',              # Web UI bare name
            'tensorneat-jax-steps-eshyperneat': 'tensorneat',  # Catalog suffixed name
            'tensorneat-compiled-eshyperneat': 'tensorneat',
            'tensorneat-optimized-eshyperneat': 'tensorneat',
            'pureples-jax': 'pureples',
        }

        # Initialize the parameter translator for unified schema support (a bidirectional translation)
        try:
            self.translator = ParameterTranslator()
            logger.info("ParameterTranslator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ParameterTranslator: {e}. Falling back to legacy mappings.")
            self.translator = None
    
    def load_config(self, algorithm: str = 'neat', implementation: str = 'pureples',
                   preset: str = 'default', config_file: Optional[str] = None,
                   overrides: Optional[Dict[str, Any]] = None, validate: bool = True,
                   use_config_params: bool = False, unified_schema: bool = False) -> Dict[str, Any]:
        """Load hierarchical configuration with proper precedence.

        Configuration precedence (highest to lowest):
        When use_config_params=False (default):
        1. Runtime overrides (overrides parameter)
        2. User config file (config_file parameter)
        3. Implementation-specific preset configuration
        4. Base algorithm defaults

        When use_config_params=True:
        1. User config file (config_file parameter)
        2. Implementation-specific preset configuration
        3. Base algorithm defaults
        (Runtime overrides are skipped - config file takes full precedence)

        Args:
            algorithm: Algorithm name (e.g., 'neat', 'hyperneat')
                      Deprecated: to be read from config.experiment.algorithm instead
            implementation: Implementation name (e.g., 'pureples', 'tensorneat')
                           Deprecated: to be read from config.experiment.implementation instead
            preset: Preset configuration name (e.g., 'default', 'high_mutation', 'conservative')
                   Deprecated: to be read from config.experiment.preset instead
            config_file: Optional path to user configuration file
            overrides: Optional dictionary of runtime overrides (ignored if use_config_params=True)
            validate: Whether to validate configuration
            use_config_params: If True, config file parameters override UI/runtime values.
                             Enables exact reproduction from saved configs.

        Returns:
            Merged configuration dictionary with metadata

        Raises:
            ConfigError: If critical configuration issues occur
            ConfigValidationError: If use_config_params=True but required params missing
        """
        # Load user config first to extract algorithm/implementation/preset if present
        user_config_data = {}
        if config_file:
            config_path = Path(config_file)
            try:
                logger.debug(f"Pre-loading user config to extract experiment parameters from {config_path}")
                user_config_data = self._load_yaml(config_path)
            except ConfigNotFoundError:
                pass  # Will be handled later
            except ConfigError:
                pass  # Will be handled later

        # Extract algorithm/implementation/preset from config if present
        # Prefer config values over function parameters
        if 'experiment' in user_config_data:
            exp_config = user_config_data['experiment']
            if 'algorithm' in exp_config:
                algorithm = exp_config['algorithm']
                logger.info(f"Using algorithm from config: {algorithm}")
            if 'implementation' in exp_config and exp_config['implementation']:
                implementation = exp_config['implementation']
                logger.info(f"Using implementation from config: {implementation}")
            if 'preset' in exp_config and exp_config['preset']:
                preset = exp_config['preset']
                logger.info(f"Using preset from config: {preset}")

        config = {}
        metadata = {
            'algorithm': algorithm,
            'implementation': implementation,
            'preset': preset,
            'loaded_files': [],
            'load_errors': [],
            'overrides_applied': bool(overrides)
        }

        # Load the implementation-specific preset configuration; the implementation-specific
        # files hold the base configs as well
        # Map implementation variant to config directory (e.g., tensorneat-compiled -> tensorneat)
        config_impl = self.impl_config_map.get(implementation, implementation)

        # Extract variant from implementation name for variant-specific config files
        # e.g., "tensorneat-compiled-eshyperneat" -> "compiled"
        # e.g., "tensorneat-jax-steps-eshyperneat" -> "jax_steps" (with underscore for filename)
        variant = None
        impl_lower = implementation.lower()
        if 'compiled' in impl_lower:
            variant = 'compiled'
        elif 'jax-steps' in impl_lower or 'jax_steps' in impl_lower:
            variant = 'jax_steps'
        elif 'optimized' in impl_lower:
            variant = 'optimized'

        # Try variant-specific config first, then fall back to generic
        # e.g., eshyperneat_compiled_default.yaml -> eshyperneat_default.yaml
        impl_config_file = None
        if variant:
            variant_config_file = self.impl_config_path / config_impl / f"{algorithm}_{variant}_{preset}.yaml"
            if variant_config_file.exists():
                impl_config_file = variant_config_file
                logger.info(f"Using variant-specific config: {variant_config_file}")
            else:
                logger.debug(f"Variant-specific config not found: {variant_config_file}, trying generic")

        # Fall back to generic config if no variant-specific found
        if impl_config_file is None:
            impl_config_file = self.impl_config_path / config_impl / f"{algorithm}_{preset}.yaml"

        try:
            if impl_config_file.exists():
                logger.debug(f"Loading implementation preset config from {impl_config_file}")
                print(f"[ConfigManager] Loading implementation: {impl_config_file}")
                config = self._load_yaml(impl_config_file)  # Direct load, no merge needed
                print(f"[ConfigManager] Config keys: {list(config.keys())}")
                metadata['loaded_files'].append(str(impl_config_file))
            else:
                error_msg = f"Implementation preset config not found: {impl_config_file}"
                logger.error(error_msg)
                metadata['load_errors'].append(error_msg)
                raise ConfigError(error_msg)
        except ConfigError as e:
            logger.error(f"Failed to load implementation preset config: {e}")
            metadata['load_errors'].append(str(e))
            raise

        # Check for deprecated JAX parameters in tensorneat section
        if 'tensorneat' in config and isinstance(config['tensorneat'], dict):
            deprecated_jax_params = ['use_jit', 'use_smart_mutations', 'use_jit_smart_mutations']
            for param in deprecated_jax_params:
                if param in config['tensorneat']:
                    logger.warning(
                        f"Deprecated: 'tensorneat.{param}' should be moved to 'performance.jax.{param}'. "
                        f"The old location will be ignored in future versions. Please update your config files."
                    )

        # 3. Load user config file if provided
        if config_file:
            config_path = Path(config_file)
            try:
                logger.debug(f"Loading user config from {config_path}")
                user_config = self._load_yaml(config_path)
                config = self._deep_merge(config, user_config)
                metadata['loaded_files'].append(str(config_path))
            except ConfigNotFoundError:
                raise  # Re-raise file not found for user configs
            except ConfigError as e:
                logger.error(f"Failed to load user config: {e}")
                raise  # Re-raise parse errors for user configs
        
        # 4. Apply runtime overrides (skip if use_config_params=True)
        if overrides and not use_config_params:
            logger.debug(f"Applying runtime overrides: {overrides}")
            print(f"[ConfigManager] Overrides keys (first 20): {list(overrides.keys())[:20]}")
            print(f"[ConfigManager] Override conn_add_prob: {overrides.get('conn_add_prob', 'NOT IN OVERRIDES')}")
            print(f"[ConfigManager] Override conn_delete_prob: {overrides.get('conn_delete_prob', 'NOT IN OVERRIDES')}")

            # Track which keys were explicitly overridden
            overridden_keys = list(overrides.keys())

            # Handle unified schema vs legacy format
            if unified_schema:
                # Grid search with unified schema - merge directly without translation
                logger.debug("[ConfigManager] Using unified schema overrides directly (no translation)")
                print("[ConfigManager] unified_schema=True - skipping parameter translation")
                hierarchical_overrides = overrides
            else:
                # Legacy format - translate from flat to hierarchical
                logger.debug("[ConfigManager] Converting legacy flat overrides to hierarchical")
                print("[ConfigManager] unified_schema=False - translating parameters")
                hierarchical_overrides = self._flat_to_hierarchical(overrides, implementation)

            logger.debug(f"Hierarchical overrides after conversion: {hierarchical_overrides}")
            print(f"[ConfigManager] Hierarchical override mutation: {hierarchical_overrides.get('mutation', 'NO MUTATION SECTION')}")
            config = self._deep_merge(config, hierarchical_overrides)
            print(f"[ConfigManager] Final mutation after overrides: add={config.get('mutation', {}).get('add_connection_prob')}, delete={config.get('mutation', {}).get('delete_connection_prob')}")
            # Store overridden keys for later use in flattening
            config['_overridden_keys'] = overridden_keys
        elif use_config_params:
            logger.info("[ConfigManager] use_config_params=True - skipping runtime overrides")
            print("[ConfigManager] use_config_params=True - config file has full precedence")

        # 5. Validate required parameters when use_config_params=True
        if use_config_params:
            self._validate_config_params(config, config_file)

        # 6. Add metadata to config
        config['_metadata'] = metadata
        config['_metadata']['use_config_params'] = use_config_params

        # 7. Validate if requested
        if validate:
            self._validate_config(config, algorithm, implementation)
        
        return config
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            ConfigNotFoundError: If file doesn't exist
            ConfigParseError: If YAML parsing fails
        """
        if not file_path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                if not isinstance(content, dict):
                    raise ConfigParseError(
                        f"Configuration file {file_path} must contain a YAML dictionary, "
                        f"got {type(content).__name__}"
                    )
                return content
        except yaml.YAMLError as e:
            raise ConfigParseError(f"Failed to parse YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigParseError(f"Error loading config file {file_path}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override the value
                result[key] = value
        
        return result

    def _validate_config_params(self, config: Dict[str, Any], config_file: Optional[str]):
        """Validate required parameters when use_config_params=True.

        Ensures config file (or defaults) contain necessary experiment parameters.

        Args:
            config: Configuration to validate
            config_file: Path to config file (for error messages)

        Raises:
            ConfigValidationError: If required parameters are missing
        """
        missing = []

        # Check for population.size
        if 'population' not in config or 'size' not in config['population']:
            missing.append("'population.size'")

        # Check for fitness.threshold (multiple possible locations)
        has_threshold = False
        if 'fitness' in config and 'threshold' in config['fitness']:
            has_threshold = True
        elif 'fitness_threshold' in config:
            has_threshold = True

        if not has_threshold:
            missing.append("'fitness.threshold' or 'fitness_threshold'")

        # generations is actually max_generations in some contexts, be flexible
        # We'll let this be optional since it can default

        if missing:
            if config_file:
                source = f"Config file '{config_file}'"
            else:
                source = "Default configuration templates"

            raise ConfigValidationError(
                f"{source} must contain {', '.join(missing)} when "
                f"'use_config_params' is enabled. Either provide a complete config file "
                f"or disable 'Use config file parameters' to use UI values."
            )

        logger.info("[ConfigManager] Config parameters validation passed")

    def _validate_config(self, config: Dict[str, Any], algorithm: str, implementation: str):
        """Validate configuration for completeness and correctness.
        
        Args:
            config: Configuration to validate
            algorithm: Algorithm name
            implementation: Implementation name
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        warnings = []
        
        # Basic validation - ensure required sections exist
        required_sections = ['population', 'mutation', 'network']
        
        for section in required_sections:
            if section not in config:
                warnings.append(f"Missing recommended section '{section}'")
                config[section] = {}
        
        # Validate population parameters
        if 'population' in config:
            pop = config['population']
            if 'size' in pop:
                if not isinstance(pop['size'], int) or pop['size'] < 1:
                    errors.append("population.size must be a positive integer")
        
        # Validate mutation parameters
        if 'mutation' in config:
            mut = config['mutation']
            prob_params = [
                'add_connection_prob', 'delete_connection_prob',
                'add_node_prob', 'delete_node_prob',
                'weight_mutate_rate', 'bias_mutate_rate'
            ]
            for param in prob_params:
                if param in mut:
                    val = mut[param]
                    if not isinstance(val, (int, float)) or val < 0 or val > 1:
                        errors.append(f"mutation.{param} must be between 0 and 1")
        
        # Ensure network dimensions are specified (info only - this is expected behavior)
        if 'num_inputs' not in config.get('network', {}):
            logger.info("network.num_inputs not specified, will be inferred from problem definition")
        if 'num_outputs' not in config.get('network', {}):
            logger.info("network.num_outputs not specified, will be inferred from problem definition")
        
        # TensorNEAT-specific warnings for connection mutation bug
        if implementation == 'tensorneat' and algorithm in ['neat', 'hyperneat']:
            self._validate_tensorneat_topology_evolution(config, warnings)
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        # Raise if errors found
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    def _validate_tensorneat_topology_evolution(self, config: Dict[str, Any], warnings: List[str]):
        """Validate TensorNEAT topology evolution parameters.
        
        Only warns when use_smart_mutations is set (off by default): nothing in this
        package or in TensorNEAT implements smart mutations.
        
        Args:
            config: Configuration to validate
            warnings: List to append warnings to
        """
        # Check if smart mutations are disabled (default: False)
        use_smart_mutations = config.get('use_smart_mutations', False)
        
        if use_smart_mutations:
            # Warn about technical debt when smart mutations are enabled
            warnings.append(
                "⚠️  WARNING: Smart mutations are enabled. This adds technical debt by mixing "
                "Python/NumPy execution with JAX's pipeline. Consider if TensorNEAT's standard "
                "2-7% mutation success rate is sufficient for your use case."
            )
        else:
            # Info about standard mutation rates when disabled
            mutation_config = config.get('mutation', {})
            conn_add_prob = mutation_config.get('add_connection_prob', 0.0)
            
            # Also check flattened parameters
            if conn_add_prob == 0.0:
                conn_add_prob = config.get('conn_add', 0.0)
            
            if conn_add_prob > 0:
                # This is info, not a warning, so we don't append to warnings
                # The user chose the recommended approach
                pass
    
    def _flat_to_hierarchical(self, flat_params: Dict[str, Any], implementation: str) -> Dict[str, Any]:
        """Convert flat parameter names to hierarchical structure.

        Uses ParameterTranslator for unified schema conversion with automatic
        deprecation warnings and duplicate handling.

        Args:
            flat_params: Flat parameter dictionary
            implementation: Implementation name ('pureples' or 'tensorneat')

        Returns:
            Hierarchical parameter dictionary
        """
        # Log input for debugging
        logger.debug(f"Converting flat params to hierarchical for {implementation}: {flat_params}")

        # Detect if input is already hierarchical (e.g., {'quadtree': {'max_weight': 8.0}})
        # If so, return as-is instead of trying to translate as flat parameters
        is_hierarchical = any(isinstance(v, dict) for v in flat_params.values())
        if is_hierarchical:
            logger.debug(f"Input is already hierarchical, returning as-is: {flat_params}")
            return flat_params

        # Use translator if available
        if self.translator:
            try:
                # Translate to unified schema
                hierarchical = self.translator.translate_from_implementation(
                    flat_params,
                    implementation,
                    algorithm_context=None  # Auto-detect from params
                )

                # Check for deprecation warnings
                if self.translator.deprecated_params:
                    for param in flat_params.keys():
                        if param in self.translator.deprecated_params:
                            deprecated_info = self.translator.deprecated_params[param]
                            logger.warning(
                                f"Deprecated parameter '{param}': "
                                f"{deprecated_info.get('reason', 'Parameter moved')}. "
                                f"Migration: {deprecated_info.get('migration', 'Update your config')}"
                            )

                logger.debug(f"Result of hierarchical conversion (via translator): {hierarchical}")
                return hierarchical
            except Exception as e:
                logger.error(f"ParameterTranslator failed: {e}. Falling back to legacy mapping.")
                # Fall through to legacy implementation

        # Legacy implementation (fallback if the translator is unavailable)
        hierarchical = {}

        # Parameter mappings to hierarchical structure
        param_to_section = {
            # Population parameters
            'population_size': ('population', 'size'),
            'pop_size': ('population', 'size'),

            # Selection parameters
            'elitism': ('selection', 'elitism'),
            'survival_threshold': ('selection', 'survival_threshold'),

            # Mutation parameters
            'conn_add_prob': ('mutation', 'add_connection_prob'),
            'conn_delete_prob': ('mutation', 'delete_connection_prob'),
            'node_add_prob': ('mutation', 'add_node_prob'),
            'node_delete_prob': ('mutation', 'delete_node_prob'),
            'weight_mutate_rate': ('mutation', 'weight_mutate_rate'),
            'bias_mutate_rate': ('mutation', 'bias_mutate_rate'),
            'activation_mutate_rate': ('mutation', 'activation_mutate_rate'),

            # Weight parameters
            'weight_mutate_power': ('weights', 'mutate_power'),
            'weight_replace_rate': ('weights', 'replace_rate'),
            'weight_init_mean': ('weights', 'init_mean'),
            'weight_init_stdev': ('weights', 'init_stdev'),
            'weight_init_std': ('weights', 'init_stdev'),
            'weight_max_value': ('weights', 'max_value'),
            'weight_min_value': ('weights', 'min_value'),

            # Network parameters
            'num_inputs': ('network', 'num_inputs'),
            'num_outputs': ('network', 'num_outputs'),
            'activation_default': ('network', 'activation_default'),
            'activation_options': ('network', 'activation_options'),

            # Fitness parameters
            'fitness_threshold': ('fitness', 'threshold'),
            'fitness_criterion': ('fitness', 'criterion'),

            # Species parameters
            'compatibility_threshold': ('population', 'compatibility_threshold'),
            'species_elitism': ('species', 'elitism'),
            'max_stagnation': ('species', 'max_stagnation'),
        }

        # Add implementation-specific mappings
        if implementation == 'tensorneat':
            tensorneat_mappings = {
                # Smart mutation parameters must go to tensorneat section
                'use_smart_mutations': ('tensorneat', 'use_smart_mutations'),
                'use_jit': ('tensorneat', 'use_jit'),
                'use_jit_smart_mutations': ('tensorneat', 'use_jit_smart_mutations'),
                'use_multiprocessing': ('tensorneat', 'use_multiprocessing'),
                'n_processes': ('tensorneat', 'n_processes'),
                'verbose': ('tensorneat', 'verbose'),
                'connection_log_frequency': ('tensorneat', 'connection_log_frequency'),
                'batch_size': ('tensorneat', 'batch_size'),
                'species_size': ('tensorneat', 'species_size'),
                'spawn_number_change_rate': ('tensorneat', 'spawn_number_change_rate'),
                'species_fitness_func': ('tensorneat', 'species_fitness_func'),
                'species_number_calculate_by': ('tensorneat', 'species_number_calculate_by'),
                # Performance settings
                'parallel_evaluations': ('performance', 'parallel_evaluations'),
                'cache_genomes': ('performance', 'cache_genomes'),
            }
            param_to_section.update(tensorneat_mappings)
            logger.debug(f"Added TensorNEAT-specific mappings for {len(tensorneat_mappings)} parameters")

        # Apply mappings
        for param, value in flat_params.items():
            if param in param_to_section:
                section, key = param_to_section[param]
                if section not in hierarchical:
                    hierarchical[section] = {}
                hierarchical[section][key] = value
                logger.debug(f"Mapped {param}={value} to {section}.{key}")
            else:
                # Keep unmapped parameters at top level
                hierarchical[param] = value
                logger.debug(f"Keeping {param}={value} at top level (no mapping)")

        # Log result for debugging
        logger.debug(f"Result of hierarchical conversion: {hierarchical}")

        return hierarchical
