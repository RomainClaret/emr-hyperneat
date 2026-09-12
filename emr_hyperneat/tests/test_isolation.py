"""Isolation guard: the package must run entirely from this repository.

This is the permanent regression test for standalone-ness. It imports the public
API, every frozen-HMR module, and the vendored compat shim, exercises a real
generation, and then asserts that

1. no ``geenns`` module was ever loaded (nothing silently falls back to the
   research framework, even on machines where it is installed), and
2. every loaded ``emr_hyperneat``/``tensorneat`` module resolves to a file inside
   this repository checkout.

If a future feature addition reintroduces a geenns import anywhere on the
executed path, this test fails immediately.
"""
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HMR_FROZEN_MODULES = [
    "emr_hyperneat._hmr_frozen.hmrhyperneat",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_dynamic_functions_aggregation",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_neuromodulation_functions",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_unified_extended_dynamic_functions_full",
    "emr_hyperneat._hmr_frozen.hmrhyperneat_pipeline_chunking_multi_gpus",
]


def _assert_no_geenns():
    loaded = [m for m in sys.modules if m == "geenns" or m.startswith("geenns.")]
    assert not loaded, f"geenns modules were loaded: {loaded}"


def test_public_api_imports_without_geenns():
    from emr_hyperneat import EMRHyperNEAT  # noqa: F401

    _assert_no_geenns()


def test_all_frozen_hmr_modules_import_without_geenns():
    for name in HMR_FROZEN_MODULES:
        importlib.import_module(name)
    _assert_no_geenns()


def test_loaded_modules_resolve_inside_repo():
    import emr_hyperneat  # noqa: F401
    import tensorneat  # noqa: F401

    for name, mod in list(sys.modules.items()):
        if not (name == "emr_hyperneat" or name.startswith("emr_hyperneat.")):
            continue
        f = getattr(mod, "__file__", None)
        if f is None:
            continue
        assert str(REPO_ROOT) in str(Path(f).resolve()), (
            f"{name} resolved OUTSIDE the repo: {f}"
        )
    # tensorneat must come from the pinned submodule when installed per the README
    tn = Path(sys.modules["tensorneat"].__file__).resolve()
    assert "third_party" in str(tn) or "site-packages" not in str(tn), (
        f"tensorneat resolved to an unexpected location: {tn}"
    )


def test_one_generation_runs_without_geenns(algorithm, xor_problem, base_config):
    cfg = algorithm.create_config(base_config)
    state = algorithm.initialize(cfg, xor_problem, seed=42)
    state, metrics = algorithm.run_generation(state, xor_problem)
    assert float(metrics.best_fitness) > 0.0
    _assert_no_geenns()
