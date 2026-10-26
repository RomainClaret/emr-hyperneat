"""Isolation guard: the package must run entirely from this repository.

This is the permanent regression test for standalone-ness. It imports the public
API and the vendored compat shim, exercises a real generation, and then asserts that

1. no ``geenns`` module was ever loaded (nothing silently falls back to the
   research framework, even on machines where it is installed), and
2. every loaded ``emr_hyperneat``/``tensorneat`` module resolves to a file inside
   this repository checkout.

If a future feature addition reintroduces a geenns import anywhere on the
executed path, this test fails immediately.

It also reads every module of the package and every script under papers/ and scripts/
by syntax tree: none may import the HMR engine (the frozen package release 0.1.0
ships, removed in 0.2.0) or pass ``backend='hmr'``, and the frozen package must
not be importable.
"""
import ast
import importlib.util
import re
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE.parent

_HMR_MODULE = re.compile(r"(^|\.)(_hmr_frozen|hmrhyperneat\w*)(\.|$)")


def _assert_no_geenns():
    loaded = [m for m in sys.modules if m == "geenns" or m.startswith("geenns.")]
    assert not loaded, f"geenns modules were loaded: {loaded}"


def test_public_api_imports_without_geenns():
    from emr_hyperneat import EMRHyperNEAT  # noqa: F401

    _assert_no_geenns()


def test_frozen_hmr_package_is_gone():
    assert importlib.util.find_spec("emr_hyperneat._hmr_frozen") is None


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


def _is_hmr(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == "hmr"


def _hmr_uses(path: Path) -> list:
    """The lines where a module imports the frozen HMR package or passes backend='hmr'.

    An import counts wherever it stands (inside a function too), and so does a call to
    import_module or __import__ with a literal name; backend='hmr' counts as a call keyword, a
    parameter default or a dict entry.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [f"{node.module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if (called in ("import_module", "__import__") and node.args
                    and isinstance(node.args[0], ast.Constant)):
                names = [str(node.args[0].value)]
            if any(k.arg == "backend" and _is_hmr(k.value) for k in node.keywords):
                found.append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            if any(a.arg == "backend" and _is_hmr(d) for a, d in pairs):
                found.append(node.lineno)
        elif isinstance(node, ast.Dict):
            if any(isinstance(k, ast.Constant) and k.value == "backend" and _is_hmr(v)
                   for k, v in zip(node.keys, node.values)):
                found.append(node.lineno)
        if any(_HMR_MODULE.search(name) for name in names):
            found.append(node.lineno)
    return found


def test_no_module_or_script_uses_hmr():
    roots = [PACKAGE, REPO_ROOT / "papers", REPO_ROOT / "scripts"]
    files = [p for root in roots if root.is_dir() for p in sorted(root.rglob("*.py"))
             if "__pycache__" not in p.parts]
    assert any(p.is_relative_to(PACKAGE) for p in files)
    users = {p.relative_to(REPO_ROOT).as_posix(): lines for p in files
             if (lines := _hmr_uses(p))}
    assert not users, users
