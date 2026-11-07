"""The configuration schemas use no API that pydantic 2 deprecates.

Every configuration passes through the vendored pydantic schemas (``_compat/schemas``) when the
engine builds TensorNEAT's settings. Pydantic 3 removes the class-based ``Config`` (deprecated since
2.0, warned when a model class is created) and reading ``model_fields`` from an instance
(deprecated since 2.11, warned at each read). ``pytest.ini`` ignores DeprecationWarning, and a
schema module imported by an earlier test has already created its classes, so each check runs in a
fresh interpreter with pydantic's deprecation warnings turned into errors.
"""
import inspect
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

TESTS = Path(__file__).resolve().parent

#: Configurations of each kind the engine accepts, built by the suite's own helpers.
CHECK_SCRIPT = r'''
import sys
import warnings
import pydantic.warnings as pw
for name in ('PydanticDeprecatedSince20', 'PydanticDeprecatedSince211'):
    if hasattr(pw, name):
        warnings.simplefilter('error', getattr(pw, name))
sys.path.insert(0, {tests!r})
from conftest import (
    EMRHyperNEAT,
    create_base_config,
    create_config_with_aggregation,
    create_config_with_dynamic_functions,
    create_config_with_leo,
    create_config_with_neuromodulation,
    create_config_with_recurrence,
)
configs = [
    create_base_config(),
    create_config_with_recurrence('hidden_only'),
    create_config_with_dynamic_functions('global_sin'),
    create_config_with_aggregation('global_max'),
    create_config_with_neuromodulation('static_gating'),
    create_config_with_leo('default'),
]
for config in configs:
    EMRHyperNEAT().create_config(config)
from emr_hyperneat._compat.core.parameter_support_validator import ParameterSupportValidator
from emr_hyperneat._compat.schemas.neat_schema import NEATConfig
ParameterSupportValidator.validate(NEATConfig(), 'tensorneat')
print('checked', len(configs))
'''


def test_create_config_raises_no_pydantic_deprecation():
    done = subprocess.run([sys.executable, '-c', CHECK_SCRIPT.format(tests=str(TESTS))],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    assert 'checked 6' in done.stdout


def test_no_schema_model_has_a_class_based_config():
    from emr_hyperneat._compat.schemas import base, hyperneat_schema, neat_schema

    models = [model for module in (base, hyperneat_schema, neat_schema)
              for _, model in inspect.getmembers(module, inspect.isclass)
              if issubclass(model, BaseModel) and model.__module__ == module.__name__]
    assert len(models) == 17
    assert [model.__name__ for model in models if 'Config' in vars(model)] == []
