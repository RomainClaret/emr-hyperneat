"""Vendored compatibility shim.

The minimal set of modules from the original geenns research framework that
EMR-HyperNEAT depends on (the abstract algorithm base, the TensorNEAT adapter/config
generator, schema models, and JAX environment helpers), copied so the algorithm runs
standalone without the geenns package. The config manager (`utils/config_manager.py`)
and its parameter translator (`config/parameter_translator.py`) are vendored but
nothing imports them: they loaded a top-level `preset` or `config_file`, which
`create_config` refuses, since none of the files they read ships. Apart from
comments, docstrings and displayed text, the copies differ from the originals in
five ways: they import from this package where the originals import from geenns,
the metrics cache path is `.emr_cache`, two modules (`adapters/metrics_storage_adapter.py`
and `core/experiment_reporter.py`) are no-op stubs, the `__init__.py` files of
the adapters, algorithms, config, core and utils sub-packages are empty, and the
internal definitions nothing here reaches are left out (most of the adapters' methods,
the translator's web-UI, duplicate and validation helpers, the unused config-manager
and precision helpers, and the experiment schema).

This is an internal implementation detail, not a public API.
"""
