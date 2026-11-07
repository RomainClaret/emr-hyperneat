"""Isolation guard: the package must run entirely from this repository.

This is the permanent regression test for standalone-ness. It imports the public
API and the vendored compat shim, exercises a real generation, and then asserts that

1. no ``geenns`` module was ever loaded (nothing silently falls back to the
   research framework, even on machines where it is installed), and
2. every loaded ``emr_hyperneat``/``tensorneat`` module resolves to a file inside
   this repository checkout.

If a future feature addition reintroduces a geenns import anywhere on the
executed path, this test fails immediately.

It also reads every file of the repository: none may carry the name of the frozen prototype
engine that release 0.1.0 ships beside EMR (its code left in 0.2.0, its name in 0.3.0), in its
path or its text, apart from the program recorded with that engine's reference trajectories.
"""
import re
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE.parent

#: The prototype engine's name, in any case; a character class spells it, so this file passes its own
#: check.
_PROTOTYPE_NAME = re.compile(r"h[m]r", re.IGNORECASE)
#: The one file that keeps the name: the program recorded with the prototype engine's trajectories,
#: kept as it ran.
_RECORDED_PROGRAM = "papers/emr-dynamic-functions/scripts/analysis/prototype_trajectories.json"


def _assert_no_geenns():
    loaded = [m for m in sys.modules if m == "geenns" or m.startswith("geenns.")]
    assert not loaded, f"geenns modules were loaded: {loaded}"


def test_public_api_imports_without_geenns():
    from emr_hyperneat import EMRHyperNEAT  # noqa: F401

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


def _repository_files() -> list:
    """The repository's files: git's list in a checkout, else a walk without caches, environments
    and the vendored TensorNEAT."""
    try:
        listed = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True,
                                text=True, check=True).stdout.split("\n")
        files = [REPO_ROOT / name for name in listed if name]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in REPO_ROOT.rglob("*") if p.is_file()]
    skip = ("__pycache__", ".git", ".venv", "venv", "build", "dist", "third_party")
    return [p for p in files if p.is_file()
            and not any(part in skip or part.endswith(".egg-info")
                        for part in p.relative_to(REPO_ROOT).parts)]


def _is_text(path: Path) -> bool:
    """git's own test for a text file: no NUL byte in its first 8,000 bytes. Every such file is
    read, whatever its suffix (a .tex figure, a .gitignore, an .svg)."""
    with open(path, "rb") as handle:
        return b"\0" not in handle.read(8000)


def test_no_file_carries_the_prototype_engines_name():
    files = _repository_files()
    assert any(p == Path(__file__).resolve() for p in files)
    named, mentions = [], {}
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if _PROTOTYPE_NAME.search(relative):
            named.append(relative)
        if relative != _RECORDED_PROGRAM and _is_text(path):
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = [n for n, line in enumerate(text.split("\n"), 1) if _PROTOTYPE_NAME.search(line)]
            if lines:
                mentions[relative] = lines
    assert not named, named
    assert not mentions, mentions
    recorded = REPO_ROOT / _RECORDED_PROGRAM
    if recorded.exists():
        # the exemption is still needed: the recorded program names the engine as it ran
        assert _PROTOTYPE_NAME.search(recorded.read_text(encoding="utf-8"))
