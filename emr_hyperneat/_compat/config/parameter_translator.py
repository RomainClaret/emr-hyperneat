"""
Parameter Translator for Config Standardization

Provides bidirectional translation between unified schema and implementation-specific
formats (TensorNEAT, PUREPLES), with context-aware routing and validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import yaml
import warnings


class ParameterTranslator:
    """
    Bidirectional parameter translator for the configuration system.

    Features:
    - Unified schema ↔ TensorNEAT/PUREPLES flat config translation
    - Web UI key ↔ unified schema mapping
    - Context-aware routing (NEAT vs HyperNEAT vs ES-HyperNEAT)
    - Duplicate parameter detection and consolidation
    - Deprecation warnings and migration guidance
    - Algorithm-specific validation

    Usage:
        translator = ParameterTranslator()

        # Web UI → Unified
        unified = {}
        translator.set_web_ui_param(unified, 'success_threshold', 0.98)

        # Unified → Implementation
        flat_config = translator.translate_to_implementation(unified, 'tensorneat')

        # Implementation → Unified
        unified = translator.translate_from_implementation(flat_config, 'tensorneat')
    """

    # Framework-only parameters that should not be translated to algorithm configs
    # These are operational/UI parameters handled by the framework itself
    FRAMEWORK_ONLY_PARAMETERS = {
        # Dashboard/Reporting - Web UI only
        'dashboard', 'dashboard_port', 'dashboard_host', 'dashboard_refresh_rate',
        'html_provider', 'no_html_optimization', 'export_formats', 'export_html',
        'telemetry_detailed_logging',

        # Variance Penalty System - Framework fitness computation
        'variance_penalty_constant', 'variance_penalty_very_low', 'variance_penalty_low',
        'variance_penalty_factor_large_constant', 'variance_penalty_factor_large_very_low',
        'variance_penalty_factor_large_low', 'variance_penalty_factor_small_constant',
        'variance_penalty_factor_small_very_low', 'variance_penalty_factor_small_low',
        'variance_penalty_thresholds', 'variance_penalty_factors_large',
        'variance_penalty_factors_small',

        # Metrics/Monitoring - Framework operational parameters
        'substrate_metrics_level', 'cppn_metrics_level', 'tensorneat_jax_metric_level',
        'lazy_metrics', 'enable_realtime_events', 'batch_device_transfers',
        'event_emission_interval', 'enable_component_profiling', 'no_subprocess_isolation',

        # Memory/Profiling - Framework managed
        'measure_memory', 'measure_jit_overhead', 'memory_limit_mb', 'max_retries',

        # Parallel Processing - Framework manages threading/multiprocessing
        'multiprocessing', 'workers', 'parallel_evaluations', 'num_genome_workers',
        'parallel_genomes',

        # JAX Operational Flags - Framework level
        'jax_memory_tracking', 'jax_memory_fraction', 'disable_smart_mutations', 'force_smart_mutations',

        # Execution Control - Framework operational behavior
        'no_fitness_termination', 'batch_evaluate',

        # PUREPLES-Specific Features - Implementation-specific operational parameters
        'fitness_sharing', 'fitness_sharing_delta',

        # Mutation Tracking - Framework analytics
        'track_mutations_detailed', 'track_mutations_analytics',
        'mutation_tracking_max_events', 'max_mutation_events',
        'track_mutation_fitness_impact', 'detailed_mutation_logging',
        'innovation_tracking', 'enable_mutation_analytics',

        # Analytics - Framework data collection
        'analytics_generation_phases', 'analytics_chain_length',
        'analytics_innovation_threshold', 'analytics_waste_threshold',

        # Innovation Tracking - Framework tracking
        'track_innovations_detailed', 'track_innovations_analytics',
        'innovation_metrics',

        # Addition: Framework metadata/operational parameters
        # These parameters configure the framework runner, not the algorithm itself
        'algorithm',              # Framework selects which algorithm to use
        'implementation',         # Framework selects which implementation
        'preset',                 # Framework config preset selection
        'trials',                 # Framework runs multiple trials
        'show_species_detail',    # Framework reporting option
    }

    def __init__(self, mappings_dir: Optional[Path] = None):
        """
        Initialize translator with mapping files.

        Args:
            mappings_dir: Directory containing mapping YAML files.
                         Defaults to config/mappings/ in this package.
        """
        if mappings_dir is None:
            mappings_dir = Path(__file__).parent / "mappings"

        self.mappings_dir = Path(mappings_dir)

        # Load all mapping files
        self.unified_to_tensorneat = self._load_mapping("unified_to_tensorneat.yaml")
        self.unified_to_pureples = self._load_mapping("unified_to_pureples.yaml")
        self.web_ui_to_unified = self._load_mapping("web_ui_to_unified.yaml")
        self.deprecated_params = self._load_mapping("deprecated_params.yaml")

        # Create reverse mappings for implementation → unified
        self.tensorneat_to_unified = self._reverse_mapping(self.unified_to_tensorneat)
        self.pureples_to_unified = self._reverse_mapping(self.unified_to_pureples)
        self.unified_to_web_ui = self._reverse_mapping(self.web_ui_to_unified)

        # Track warnings issued to avoid duplicates
        self._issued_warnings = set()

    def _load_mapping(self, filename: str) -> Dict:
        """Load mapping file from mappings directory."""
        filepath = self.mappings_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {filepath}\n"
                f"Expected in: {self.mappings_dir}"
            )

        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}

    def _reverse_mapping(self, mapping: Dict) -> Dict:
        """
        Create reverse mapping (flat key → unified path).

        Handles context-dependent mappings by flattening context variants.

        Args:
            mapping: Forward mapping (unified.path → flat_key or {context: flat_key})

        Returns:
            Reverse mapping (flat_key → unified.path)
        """
        reverse = {}

        for unified_path, flat_key in mapping.items():
            if isinstance(flat_key, dict):
                # Context-dependent mapping - create entries for all contexts
                for context, context_key in flat_key.items():
                    if context_key not in reverse:
                        reverse[context_key] = []
                    reverse[context_key].append({
                        'unified_path': unified_path,
                        'context': context
                    })
            else:
                # Direct mapping
                if flat_key not in reverse:
                    reverse[flat_key] = []
                reverse[flat_key].append({
                    'unified_path': unified_path,
                    'context': None
                })

        return reverse

    # =========================================================================
    # CONTEXT DETECTION
    # =========================================================================

    def detect_algorithm_context(self, params: Dict) -> str:
        """
        Detect algorithm type from parameters.

        Logic:
        1. Has quadtree.* section → 'es_hyperneat'
        2. Has substrate.* OR cppn.* OR genome.type='cppn' → 'hyperneat'
        3. Otherwise → 'neat'

        Args:
            params: Either hierarchical unified params or flat implementation params

        Returns:
            'neat', 'hyperneat', or 'es_hyperneat'
        """
        # Check for quadtree (ES-HyperNEAT)
        if 'quadtree' in params:
            return 'es_hyperneat'

        # Check for HyperNEAT indicators
        has_substrate = 'substrate' in params
        has_cppn = 'cppn' in params

        # Check genome.type (handles both nested and flat configs)
        genome_type_cppn = False
        if 'genome' in params and isinstance(params['genome'], dict):
            genome_type_cppn = params['genome'].get('type') == 'cppn'
        elif 'genome_type' in params:
            genome_type_cppn = params['genome_type'] == 'cppn'

        if has_substrate or has_cppn or genome_type_cppn:
            return 'hyperneat'

        # Check flat config keys for HyperNEAT/ES-HyperNEAT
        flat_keys = set(params.keys())
        hyperneat_keys = {
            'substrate_dimensions', 'substrate_input_nodes', 'cppn_num_inputs',
            'cppn_activation_default', 'weight_threshold', 'max_weight'
        }
        es_hyperneat_keys = {
            'initial_depth', 'max_depth', 'variance_threshold',
            'division_threshold', 'band_threshold'
        }

        if flat_keys & es_hyperneat_keys:
            return 'es_hyperneat'
        if flat_keys & hyperneat_keys:
            return 'hyperneat'

        return 'neat'

    # =========================================================================
    # WEB UI INTEGRATION
    # =========================================================================

    def get_web_ui_param(self, unified_params: Dict, web_key: str) -> Any:
        """
        Extract value using Web UI key from unified params.

        Args:
            unified_params: Hierarchical unified params
            web_key: Frontend parameter key (e.g., 'success_threshold')

        Returns:
            Parameter value or None if not found

        Example:
            value = translator.get_web_ui_param(unified, 'success_threshold')
            # Extracts from unified['fitness']['threshold']
        """
        if web_key not in self.web_ui_to_unified:
            return None

        unified_path = self.web_ui_to_unified[web_key]
        return self._get_nested_value(unified_params, unified_path)

    def set_web_ui_param(self, unified_params: Dict, web_key: str, value: Any) -> Dict:
        """
        Set value using Web UI key in unified params.

        Args:
            unified_params: Hierarchical unified params (modified in place)
            web_key: Frontend parameter key (e.g., 'success_threshold')
            value: Parameter value to set

        Returns:
            Modified unified_params

        Example:
            translator.set_web_ui_param(unified, 'success_threshold', 0.98)
            # Sets unified['fitness']['threshold'] = 0.98
        """
        # Check if parameter is framework-only (should not be translated to algorithm config)
        if web_key in self.FRAMEWORK_ONLY_PARAMETERS:
            # Store in special framework section for traceability, but don't warn
            if 'framework' not in unified_params:
                unified_params['framework'] = {}
            unified_params['framework'][web_key] = value
            return unified_params

        if web_key not in self.web_ui_to_unified:
            warnings.warn(
                f"Unknown Web UI parameter '{web_key}' - not in mapping table. "
                f"Parameter will be ignored."
            )
            return unified_params

        unified_path = self.web_ui_to_unified[web_key]
        self._set_nested_value(unified_params, unified_path, value)
        return unified_params

    # =========================================================================
    # IMPLEMENTATION TRANSLATION
    # =========================================================================

    def translate_to_implementation(
        self,
        unified_params: Dict,
        impl: str,
        algorithm_context: Optional[str] = None
    ) -> Dict:
        """
        Convert unified schema → implementation-specific flat dict.

        Args:
            unified_params: Hierarchical unified schema params
            impl: 'tensorneat' or 'pureples'
            algorithm_context: 'neat', 'hyperneat', or 'es_hyperneat'
                              If None, will be auto-detected

        Returns:
            Flat implementation-specific parameter dict

        Example:
            unified = {'fitness': {'threshold': 0.98}, 'population': {'size': 150}}
            flat = translator.translate_to_implementation(unified, 'tensorneat')
            # Returns: {'fitness_threshold': 0.98, 'population_size': 150}
        """
        if impl not in ['tensorneat', 'pureples']:
            raise ValueError(f"Invalid implementation: {impl}. Must be 'tensorneat' or 'pureples'")

        # Auto-detect context if not provided
        if algorithm_context is None:
            algorithm_context = self.detect_algorithm_context(unified_params)

        # Select appropriate mapping
        mapping = (self.unified_to_tensorneat if impl == 'tensorneat'
                  else self.unified_to_pureples)

        flat_config = {}

        # Flatten unified params and translate each
        flattened = self._flatten_dict(unified_params)

        for unified_path, value in flattened.items():
            # Skip implementation-specific sections (they pass through)
            # Also skip framework-only parameters (operational, not algorithmic)
            if (unified_path.startswith('tensorneat.') or
                unified_path.startswith('pureples.') or
                unified_path.startswith('framework.')):
                # Strip prefix and add to flat config (but skip framework params)
                if not unified_path.startswith('framework.'):
                    flat_key = unified_path.split('.', 1)[1] if '.' in unified_path else unified_path
                    flat_config[flat_key] = value
                continue

            # Translate using mapping
            if unified_path in mapping:
                flat_key = mapping[unified_path]

                # Handle context-dependent mappings
                if isinstance(flat_key, dict):
                    if algorithm_context in flat_key:
                        flat_key = flat_key[algorithm_context]
                    else:
                        # Context not applicable, skip this parameter
                        continue

                flat_config[flat_key] = value
            else:
                # No mapping found - may be algorithm-specific or new parameter
                # Issue warning but don't fail
                warning_key = f"unmapped_unified_{impl}_{unified_path}"
                if warning_key not in self._issued_warnings:
                    warnings.warn(
                        f"No mapping for unified parameter '{unified_path}' → {impl}. "
                        f"Parameter will be omitted from flat config."
                    )
                    self._issued_warnings.add(warning_key)

        return flat_config

    def translate_from_implementation(
        self,
        impl_params: Dict,
        impl: str,
        algorithm_context: Optional[str] = None
    ) -> Dict:
        """
        Convert implementation-specific flat → unified hierarchical.

        Args:
            impl_params: Flat implementation params
            impl: 'tensorneat' or 'pureples'
            algorithm_context: If None, will be auto-detected

        Returns:
            Hierarchical unified schema params

        Example:
            flat = {'fitness_threshold': 0.98, 'population_size': 150}
            unified = translator.translate_from_implementation(flat, 'tensorneat')
            # Returns: {'fitness': {'threshold': 0.98}, 'population': {'size': 150}}
        """
        if impl not in ['tensorneat', 'pureples']:
            raise ValueError(f"Invalid implementation: {impl}. Must be 'tensorneat' or 'pureples'")

        # Auto-detect context if not provided
        if algorithm_context is None:
            algorithm_context = self.detect_algorithm_context(impl_params)

        # Select appropriate reverse mapping
        reverse_mapping = (self.tensorneat_to_unified if impl == 'tensorneat'
                          else self.pureples_to_unified)

        unified_params = {}

        for flat_key, value in impl_params.items():
            if flat_key in reverse_mapping:
                # Get all possible unified paths for this flat key
                mappings = reverse_mapping[flat_key]

                # Find the right mapping based on context
                unified_path = None
                for mapping_info in mappings:
                    if mapping_info['context'] is None or mapping_info['context'] == algorithm_context:
                        unified_path = mapping_info['unified_path']
                        break

                if unified_path:
                    self._set_nested_value(unified_params, unified_path, value)
            else:
                # No mapping - may be implementation-specific parameter
                # Store in implementation-specific section
                impl_section = 'tensorneat' if impl == 'tensorneat' else 'pureples'
                if impl_section not in unified_params:
                    unified_params[impl_section] = {}
                unified_params[impl_section][flat_key] = value

        return unified_params

    # =========================================================================
    # DUPLICATE HANDLING
    # =========================================================================

    def detect_duplicates(self, params: Dict) -> Dict[str, List[str]]:
        """
        Find duplicate parameters across sections.

        Identifies parameters that appear in multiple locations (e.g., top-level
        'weights' and 'cppn.weights') that should be consolidated.

        Args:
            params: Hierarchical config params (may have old structure with duplicates)

        Returns:
            Dict mapping canonical_path → [duplicate_path1, duplicate_path2, ...]

        Example:
            duplicates = translator.detect_duplicates(config)
            # Returns: {
            #   'network.connections.weights.init_mean': [
            #     'weights.init_mean',
            #     'cppn.weights.init_mean'
            #   ]
            # }
        """
        duplicates = {}
        flattened = self._flatten_dict(params)

        for old_path in flattened.keys():
            if old_path in self.deprecated_params:
                deprecated_info = self.deprecated_params[old_path]
                new_path = deprecated_info.get('new_path')

                if new_path:
                    if new_path not in duplicates:
                        duplicates[new_path] = []
                    duplicates[new_path].append(old_path)

        # Only return entries with actual duplicates
        return {k: v for k, v in duplicates.items() if len(v) > 0}

    def consolidate_duplicates(
        self,
        params: Dict,
        conflict_resolution: str = 'warn'
    ) -> Tuple[Dict, List[str]]:
        """
        Consolidate duplicate parameters to canonical paths.

        When multiple deprecated paths map to same canonical path, consolidates
        to single location. Handles conflicts based on resolution strategy.

        Args:
            params: Hierarchical config with potential duplicates
            conflict_resolution: How to handle value conflicts:
                - 'warn': Use first value, warn about conflicts
                - 'error': Raise exception on conflicts
                - 'first': Silently use first value
                - 'last': Silently use last value

        Returns:
            (consolidated_params, warnings_list)

        Example:
            config = {
                'weights': {'init_mean': 0.0},
                'cppn': {'weights': {'init_mean': 0.5}}
            }
            consolidated, warnings = translator.consolidate_duplicates(config)
            # Returns: (
            #   {'network': {'connections': {'weights': {'init_mean': 0.0}}}},
            #   ["CONFLICT: weights.init_mean (0.0) vs cppn.weights.init_mean (0.5)"]
            # )
        """
        warnings_list = []
        consolidated = {}

        # Detect all duplicates
        duplicates = self.detect_duplicates(params)
        flattened = self._flatten_dict(params)

        # Track which old paths have been processed
        processed_old_paths = set()

        # Process each canonical path
        for canonical_path, old_paths in duplicates.items():
            values = []

            # Collect all values from old paths
            for old_path in old_paths:
                if old_path in flattened:
                    values.append((old_path, flattened[old_path]))
                    processed_old_paths.add(old_path)

            if not values:
                continue

            # Check for conflicts (handle unhashable types like lists)
            def make_hashable(v):
                """Convert value to hashable type for comparison."""
                if isinstance(v, list):
                    return tuple(v)
                elif isinstance(v, dict):
                    return tuple(sorted(v.items()))
                return v

            try:
                unique_values = set(make_hashable(v) for _, v in values if v is not None)
            except TypeError:
                # If still unhashable, compare by string representation
                unique_values = set(str(v) for _, v in values if v is not None)

            if len(unique_values) > 1:
                # Conflict detected
                conflict_msg = (
                    f"CONFLICT: Multiple values for '{canonical_path}': " +
                    ", ".join(f"{path}={val}" for path, val in values)
                )

                if conflict_resolution == 'error':
                    raise ValueError(conflict_msg)
                elif conflict_resolution == 'warn':
                    warnings_list.append(conflict_msg)
                    # Use first non-None value
                    canonical_value = next(v for _, v in values if v is not None)
                elif conflict_resolution == 'first':
                    canonical_value = next(v for _, v in values if v is not None)
                elif conflict_resolution == 'last':
                    canonical_value = next(v for _, v in reversed(values) if v is not None)
                else:
                    raise ValueError(f"Invalid conflict_resolution: {conflict_resolution}")
            else:
                # No conflict, use the value
                canonical_value = next(iter(unique_values))

            # Set canonical value
            self._set_nested_value(consolidated, canonical_path, canonical_value)

            # Add deprecation warning
            deprecated_info = self.deprecated_params.get(old_paths[0], {})
            if deprecated_info and len(old_paths) > 0:
                warnings_list.append(
                    f"DEPRECATED: {old_paths[0]} → {canonical_path}. "
                    f"Reason: {deprecated_info.get('reason', 'Consolidated parameter')}. "
                    f"Migration: {deprecated_info.get('migration', 'Update your config')}"
                )

        # Copy over non-duplicate parameters
        for flat_path, value in flattened.items():
            if flat_path not in processed_old_paths:
                # Check if this path is deprecated (but not part of duplicates)
                if flat_path in self.deprecated_params:
                    deprecated_info = self.deprecated_params[flat_path]
                    new_path = deprecated_info.get('new_path')
                    if new_path:
                        self._set_nested_value(consolidated, new_path, value)
                        warnings_list.append(
                            f"DEPRECATED: {flat_path} → {new_path}. "
                            f"Migration: {deprecated_info.get('migration', 'Update your config')}"
                        )
                    else:
                        # No new path specified, copy as-is
                        self._set_nested_value(consolidated, flat_path, value)
                else:
                    # Not deprecated, copy as-is
                    self._set_nested_value(consolidated, flat_path, value)

        return consolidated, warnings_list

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_unified(
        self,
        unified_params: Dict,
        algorithm_context: Optional[str] = None
    ) -> List[str]:
        """
        Validate unified params against schema and algorithm context.

        Checks:
        - Required parameters for algorithm type (NEAT/HyperNEAT/ES-HyperNEAT)
        - Invalid parameters for algorithm type
        - Value constraints and types

        Args:
            unified_params: Hierarchical unified params
            algorithm_context: If None, will be auto-detected

        Returns:
            List of validation error messages (empty if valid)

        Example:
            errors = translator.validate_unified(config)
            if errors:
                print("Validation failed:")
                for error in errors:
                    print(f"  - {error}")
        """
        errors = []

        # Auto-detect context if not provided
        if algorithm_context is None:
            algorithm_context = self.detect_algorithm_context(unified_params)

        # NEAT-specific validation
        if algorithm_context == 'neat':
            # Must have network topology
            if 'network' not in unified_params or 'topology' not in unified_params.get('network', {}):
                errors.append("NEAT requires 'network.topology' section")
            else:
                topology = unified_params['network']['topology']
                if 'num_inputs' not in topology:
                    errors.append("NEAT requires 'network.topology.num_inputs'")
                if 'num_outputs' not in topology:
                    errors.append("NEAT requires 'network.topology.num_outputs'")

            # Should NOT have HyperNEAT sections
            if 'substrate' in unified_params:
                errors.append("NEAT should not have 'substrate' section (HyperNEAT only)")
            if 'cppn' in unified_params:
                errors.append("NEAT should not have 'cppn' section (HyperNEAT only)")
            if 'quadtree' in unified_params:
                errors.append("NEAT should not have 'quadtree' section (ES-HyperNEAT only)")

        # HyperNEAT-specific validation
        elif algorithm_context == 'hyperneat':
            # Must have substrate and cppn
            if 'substrate' not in unified_params:
                errors.append("HyperNEAT requires 'substrate' section")
            if 'cppn' not in unified_params:
                errors.append("HyperNEAT requires 'cppn' section")

            # Validate substrate coordinates match layer nodes
            if 'substrate' in unified_params:
                substrate = unified_params['substrate']
                if 'layers' in substrate and 'input_coordinates' in substrate:
                    input_nodes = substrate.get('layers', {}).get('input', {}).get('nodes')
                    input_coords = substrate.get('input_coordinates', [])
                    if input_nodes and len(input_coords) != input_nodes:
                        errors.append(
                            f"Substrate input_coordinates length ({len(input_coords)}) "
                            f"must match substrate.layers.input.nodes ({input_nodes})"
                        )

            # Validate CPPN inputs (4 for 2D, 5 for 3D)
            if 'cppn' in unified_params:
                cppn = unified_params['cppn']
                num_inputs = cppn.get('network', {}).get('num_inputs')
                if num_inputs and num_inputs not in [4, 5]:
                    errors.append(
                        f"CPPN num_inputs must be 4 (2D) or 5 (3D), got {num_inputs}"
                    )

        # ES-HyperNEAT-specific validation
        elif algorithm_context == 'es_hyperneat':
            # Must have all HyperNEAT requirements plus quadtree
            if 'substrate' not in unified_params:
                errors.append("ES-HyperNEAT requires 'substrate' section")
            if 'cppn' not in unified_params:
                errors.append("ES-HyperNEAT requires 'cppn' section")
            if 'quadtree' not in unified_params:
                errors.append("ES-HyperNEAT requires 'quadtree' section")

            # Validate quadtree constraints
            if 'quadtree' in unified_params:
                quadtree = unified_params['quadtree']
                initial_depth = quadtree.get('initial_depth')
                max_depth = quadtree.get('max_depth')

                if initial_depth is not None and max_depth is not None:
                    if max_depth <= initial_depth:
                        errors.append(
                            f"Quadtree max_depth ({max_depth}) must be > "
                            f"initial_depth ({initial_depth})"
                        )

        return errors

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_nested_value(self, data: Dict, path: str, default: Any = None) -> Any:
        """
        Get value from nested dict using dot-separated path.

        Args:
            data: Nested dict
            path: Dot-separated path (e.g., 'fitness.threshold')
            default: Default value if path not found

        Returns:
            Value at path or default
        """
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def _set_nested_value(self, data: Dict, path: str, value: Any) -> None:
        """
        Set value in nested dict using dot-separated path.

        Creates intermediate dicts as needed.

        Args:
            data: Nested dict (modified in place)
            path: Dot-separated path (e.g., 'fitness.threshold')
            value: Value to set
        """
        keys = path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _flatten_dict(self, data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dict to dot-separated keys.

        Args:
            data: Nested dict
            parent_key: Prefix for keys (used in recursion)
            sep: Separator between keys

        Returns:
            Flattened dict with dot-separated keys

        Example:
            nested = {'fitness': {'threshold': 0.98}}
            flat = _flatten_dict(nested)
            # Returns: {'fitness.threshold': 0.98}
        """
        items = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))

        return dict(items)
