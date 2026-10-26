"""Single source of truth for the package version.

Kept in its own module with no imports so setuptools can read it statically
(``[tool.setuptools.dynamic]`` in pyproject.toml) without importing JAX.
"""

__version__ = "0.2.0"
