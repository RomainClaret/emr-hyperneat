"""
Parameter Translator for Config Standardization

Translation from implementation-specific formats (TensorNEAT, PUREPLES) to the unified
schema, with context-aware routing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ParameterTranslator:
    """
    Parameter translator for the configuration system: TensorNEAT or PUREPLES flat
    configurations to the unified schema, with context-aware routing (NEAT, HyperNEAT,
    ES-HyperNEAT) and the deprecated-parameter table.

    The mapping files it reads are not part of this package, so constructing it raises
    FileNotFoundError here and ConfigManager runs without a translator.

    Usage:
        translator = ParameterTranslator(mappings_dir)
        unified = translator.translate_from_implementation(flat_config, 'tensorneat')
    """

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

        # Create reverse mappings for implementation -> unified
        self.tensorneat_to_unified = self._reverse_mapping(self.unified_to_tensorneat)
        self.pureples_to_unified = self._reverse_mapping(self.unified_to_pureples)
        self.unified_to_web_ui = self._reverse_mapping(self.web_ui_to_unified)

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
        Create reverse mapping (flat key -> unified path).

        Handles context-dependent mappings by flattening context variants.

        Args:
            mapping: Forward mapping (unified.path -> flat_key or {context: flat_key})

        Returns:
            Reverse mapping (flat_key -> unified.path)
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

    # Context detection

    def detect_algorithm_context(self, params: Dict) -> str:
        """
        Detect algorithm type from parameters.

        Logic:
        1. Has quadtree.* section -> 'es_hyperneat'
        2. Has substrate.* OR cppn.* OR genome.type='cppn' -> 'hyperneat'
        3. Otherwise -> 'neat'

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

    # Implementation translation

    def translate_from_implementation(
        self,
        impl_params: Dict,
        impl: str,
        algorithm_context: Optional[str] = None
    ) -> Dict:
        """
        Convert implementation-specific flat -> unified hierarchical.

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

    # Utility methods

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
