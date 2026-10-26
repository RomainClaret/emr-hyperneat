"""
Parameter Support Validator

Validates that configuration parameters are supported by the target implementation
by inspecting json_schema_extra metadata added via impl_meta() helper.

This validator provides warnings (not errors) when users configure parameters that
aren't supported by their chosen NEAT implementation (PUREPLES vs TensorNEAT).

Example:
    >>> from emr_hyperneat._compat.schemas.neat_schema import NEATConfig
    >>> config = NEATConfig(genome={'enabled_mutate_rate': 0.5})  # PUREPLES-only
    >>> result = ParameterSupportValidator.validate(config, 'tensorneat')
    >>> for warning in result.warnings:
    ...     print(warning)
    Parameter 'enabled_mutate_rate' = 0.5 not supported by tensorneat...
"""

import logging
from dataclasses import dataclass, field
from typing import List, Any, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of parameter support validation.

    Attributes:
        valid: Always True for warnings-only validation
        errors: List of error messages (empty for warnings-only mode)
        warnings: List of warning messages for unsupported parameters
        implementation: Target implementation name
        checked_parameters: Number of parameters checked
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    implementation: str = ""
    checked_parameters: int = 0

    def log_warnings(self) -> None:
        """Log all warnings using Python logging."""
        for warning in self.warnings:
            logger.warning(warning)


class ParameterSupportValidator:
    """Validates configuration parameters against implementation support metadata.

    This validator inspects the json_schema_extra field added by impl_meta() to
    determine which parameters are supported by which implementations.

    Validation Strategy:
        - Warnings only: Unsupported parameters generate warnings, not errors
        - Non-default values only: Only warn if user explicitly set a value
        - Recursive validation: Walks nested config structures
        - Clear messages: Each warning includes parameter name, value, and supported implementations

    Usage:
        result = ParameterSupportValidator.validate(config, 'tensorneat')
        result.log_warnings()  # Log via Python logging

        # Or check manually
        if result.has_issues():
            for warning in result.warnings:
                print(warning)
    """

    @staticmethod
    def validate(
        config: Union[BaseModel, Any],
        implementation: str,
        path: str = ""
    ) -> ValidationResult:
        """Validate configuration parameters against implementation support.

        Args:
            config: Pydantic configuration model (NEATConfig, HyperNEATConfig, etc.)
            implementation: Target implementation name ('pureples', 'tensorneat', etc.)
            path: Current path in config tree (used for recursion, leave empty)

        Returns:
            ValidationResult with warnings for unsupported parameters

        Example:
            >>> config = NEATConfig()
            >>> config.genome.enabled_mutate_rate = 0.5  # PUREPLES-only
            >>> result = ParameterSupportValidator.validate(config, 'tensorneat')
            >>> assert len(result.warnings) == 1
            >>> assert 'enabled_mutate_rate' in result.warnings[0]
        """
        result = ValidationResult(implementation=implementation)

        # Only validate Pydantic models
        if not isinstance(config, BaseModel):
            return result

        # Walk all fields in the model
        for field_name, field_info in config.model_fields.items():
            current_path = f"{path}.{field_name}" if path else field_name
            current_value = getattr(config, field_name, None)

            # Special enforcement: TensorNEAT aggregation='sum' limitation
            if implementation == 'tensorneat' and current_path.endswith('aggregation'):
                ParameterSupportValidator._validate_tensorneat_aggregation(
                    current_value, current_path, result
                )

            # Check if field has implementation support metadata
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra

                if isinstance(extra, dict) and 'implementation_support' in extra:
                    support_info = extra['implementation_support']
                    supported_by = set(support_info.get('supported_by', []))
                    notes = support_info.get('notes')

                    result.checked_parameters += 1

                    # Check if current implementation is supported
                    if implementation not in supported_by:
                        # Only warn if value differs from default
                        if current_value != field_info.default:
                            warning_msg = (
                                f"Parameter '{current_path}' = {current_value} "
                                f"is not supported by '{implementation}'. "
                                f"Supported by: {sorted(supported_by)}. "
                                f"This parameter will be ignored."
                            )

                            # Add implementation-specific notes if available
                            if notes:
                                warning_msg += f" Note: {notes}"

                            result.warnings.append(warning_msg)

            # Recursively validate nested BaseModel fields
            if isinstance(current_value, BaseModel):
                nested_result = ParameterSupportValidator.validate(
                    current_value,
                    implementation,
                    current_path
                )
                result.warnings.extend(nested_result.warnings)
                result.errors.extend(nested_result.errors)
                result.checked_parameters += nested_result.checked_parameters
                # Propagate validation failure
                if not nested_result.valid:
                    result.valid = False

        return result

    @staticmethod
    def _validate_tensorneat_aggregation(
        aggregation_config: Any,
        path: str,
        result: ValidationResult
    ) -> None:
        """Enforce TensorNEAT's aggregation='sum' limitation.

        TensorNEAT only supports 'sum' aggregation. This is a critical constraint
        that must raise an error, not just a warning, since using other aggregation
        functions will cause runtime failures.

        Args:
            aggregation_config: AggregationConfig instance to validate
            path: Current path for error messages
            result: ValidationResult to update with errors
        """
        if not isinstance(aggregation_config, BaseModel):
            return

        # Check default aggregation function
        default_value = getattr(aggregation_config, 'default', None)
        if default_value and default_value != 'sum':
            result.valid = False
            result.errors.append(
                f"TensorNEAT aggregation constraint violated at '{path}.default': "
                f"TensorNEAT only supports aggregation='sum', but '{default_value}' was specified. "
                f"Change to 'sum' or use PUREPLES implementation for other aggregation functions."
            )

        # Check options list
        options = getattr(aggregation_config, 'options', None)
        if options and options != ['sum']:
            result.valid = False
            result.errors.append(
                f"TensorNEAT aggregation constraint violated at '{path}.options': "
                f"TensorNEAT only supports aggregation=['sum'], but {options} was specified. "
                f"Change to ['sum'] or use PUREPLES implementation for other aggregation functions."
            )
