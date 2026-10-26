#!/usr/bin/env python3
"""Run a paper's experiments on the emr-hyperneat release it was published with.

Each paper under ``papers/`` names its release in ``papers/releases.json``. This script builds that
release in its own checkout, ``.replicate/v<release>/``: a git worktree of the release tag with the
TensorNEAT submodule and a virtual environment installed as ``docs/installation.md`` describes
(about 2 GB). Commands then run there, so the current checkout and environment are untouched.

Usage
-----
    python scripts/replicate.py list
    python scripts/replicate.py setup emr-dynamic-functions
    python scripts/replicate.py run emr-hyperneat -- python scripts/analysis/compute_speedup.py
    python scripts/replicate.py run --release 0.1.0 -- python scripts/fetch_results.py
    python scripts/replicate.py run emr-dynamic-functions --release current -- python ...

With a paper, ``run`` executes the command in that paper's folder of the release checkout (the paper
READMEs give their commands from there); with only ``--release``, at the checkout's root. It sets
the release up first when needed. ``--release`` picks another release than the paper's, and
``--release current`` runs in this checkout, with the Python that runs this script first on PATH.

Remove a release checkout with ``git worktree remove --force .replicate/v<release>``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASES_FILE = REPO_ROOT / "papers" / "releases.json"
CURRENT = "current"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_BIN = "Scripts" if os.name == "nt" else "bin"
_MARKER = "replicate-ok"


def load_releases() -> dict[str, str]:
    """Return the paper -> release map of ``papers/releases.json``."""
    return json.loads(RELEASES_FILE.read_text())


def current_version() -> str:
    """Return the version of this checkout's package, read from ``_version.py``."""
    text = (REPO_ROOT / "emr_hyperneat" / "_version.py").read_text()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def normalize(release: str) -> str:
    """Return ``release`` as ``X.Y.Z`` (a leading ``v`` is dropped), or ``current``."""
    if release == CURRENT:
        return release
    version = release[1:] if release.startswith("v") else release
    if not _VERSION_RE.match(version):
        raise SystemExit(f"Not a release: {release!r} (expected e.g. 0.1.0, v0.1.0 or {CURRENT}).")
    return version


def resolve(paper: Optional[str], release: Optional[str], releases: dict[str, str]) -> str:
    """Return the release to use: ``release`` if given, else the paper's own."""
    if paper is not None and paper not in releases:
        raise SystemExit(f"Unknown paper {paper!r}. Papers: {', '.join(sorted(releases))}.")
    if release is not None:
        return normalize(release)
    if paper is None:
        raise SystemExit("Name a paper, or a release with --release.")
    return releases[paper]


def _git(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("git is required: the releases are git tags of this repository.")


def tag_commit(version: str) -> str:
    """Return the commit of tag ``v<version>``, or exit with how to get it."""
    if _git("rev-parse", "--git-dir").returncode != 0:
        raise SystemExit(
            "This is not a git clone, so the release tags are not here. Clone the repository with\n"
            "  git clone --recursive https://github.com/RomainClaret/emr-hyperneat.git"
        )
    found = _git("rev-parse", "--verify", "--quiet", f"refs/tags/v{version}^{{commit}}")
    if found.returncode != 0:
        raise SystemExit(f"Tag v{version} is not in this clone. Fetch the tags with\n"
                         f"  git fetch --tags origin")
    return found.stdout.strip()


def checkout_dir(version: str) -> Path:
    """Return the release checkout's directory, ``.replicate/v<version>``."""
    return REPO_ROOT / ".replicate" / f"v{version}"


def venv_bin(checkout: Path) -> Path:
    """Return the ``bin`` (``Scripts`` on Windows) directory of a checkout's environment."""
    return checkout / ".venv" / _BIN


def build_env(checkout: Path, python: str) -> None:
    """Create the checkout's environment and install it as ``docs/installation.md`` describes."""
    major_minor = subprocess.run(
        [python, "-c", "import sys; print(*sys.version_info[:2])"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not (3, 10) <= tuple(map(int, major_minor)) <= (3, 12):
        raise SystemExit(
            f"{python} is Python {'.'.join(major_minor)}; the releases' pinned stack (NumPy "
            f"1.26.4)\ninstalls on Python 3.10 to 3.12. Pass one with --python, e.g. "
            f"--python python3.12."
        )
    subprocess.run([python, "-m", "venv", str(checkout / ".venv")], check=True)
    py = str(venv_bin(checkout) / "python")
    steps = [
        ["-m", "pip", "install", "--upgrade", "pip"],
        ["-m", "pip", "install", "-e", ".", "-c", "requirements-lock.txt"],
        ["-m", "pip", "install", "-e", "third_party/tensorneat", "-c", "requirements-lock.txt"],
        ["-m", "pip", "install", "-e", ".[analysis,dev]"],
    ]
    for step in steps:
        subprocess.run([py, *step], cwd=checkout, check=True)


def _check_env(checkout: Path, version: str) -> None:
    code = ("import os, emr_hyperneat as e; "
            "print(e.__version__); print(os.path.realpath(e.__file__))")
    probe = subprocess.run([str(venv_bin(checkout) / "python"), "-c", code],
                           cwd=checkout, capture_output=True, text=True)
    lines = probe.stdout.split("\n") + ["", ""]
    if (probe.returncode != 0 or lines[0] != version
            or not lines[1].startswith(str(checkout.resolve()))):
        raise SystemExit(f"The environment in {checkout} does not import emr_hyperneat {version} "
                         f"from that checkout:\n{probe.stdout}{probe.stderr}")


def setup(version: str, python: str = sys.executable,
          make_env: Callable[[Path, str], None] = build_env, check: bool = True) -> Path:
    """Build release ``version`` in ``.replicate/v<version>`` once, and return that directory."""
    commit = tag_commit(version)
    checkout = checkout_dir(version)
    if checkout.exists():
        top = _git("rev-parse", "--show-toplevel", cwd=checkout).stdout.strip()
        head = _git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
        if Path(top or "/").resolve() != checkout.resolve() or head != commit:
            raise SystemExit(f"{checkout} is not a checkout of v{version}. Remove it with\n"
                             f"  git worktree remove --force {checkout}")
    else:
        checkout.parent.mkdir(exist_ok=True)
        _git("worktree", "prune")  # forget checkouts whose directory was deleted by hand
        added = _git("worktree", "add", "--detach", str(checkout), f"v{version}")
        if added.returncode != 0:
            raise SystemExit(f"git worktree add failed:\n{added.stderr}")
    if _git("submodule", "update", "--init", "--recursive", cwd=checkout).returncode != 0:
        raise SystemExit(f"git submodule update failed in {checkout}; it needs network access "
                         f"to fetch TensorNEAT.")
    marker = checkout / ".venv" / _MARKER
    if not (marker.exists() and marker.read_text().strip() == commit):
        print(f"Installing release {version} in {checkout} (about 2 GB, a few minutes)...")
        try:
            make_env(checkout, python)
        except subprocess.CalledProcessError as err:
            raise SystemExit(f"Installing release {version} failed at: {' '.join(err.cmd)}")
        if check:
            _check_env(checkout, version)
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(commit + "\n")
    return checkout


def run(paper: Optional[str], version: str, command: list[str], python: str = sys.executable,
        releases: Optional[dict[str, str]] = None) -> int:
    """Run ``command`` on release ``version``, in ``paper``'s folder if a paper is named."""
    releases = load_releases() if releases is None else releases
    if not command:
        raise SystemExit("Give the command after --, e.g. -- python scripts/fetch_results.py")
    if version == CURRENT:
        root, env, used = REPO_ROOT, dict(os.environ), current_version()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    else:
        root, used = setup(version, python), version
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["VIRTUAL_ENV"] = str(root / ".venv")
        env["PATH"] = str(venv_bin(root)) + os.pathsep + env.get("PATH", "")
    cwd = root / "papers" / paper if paper else root
    if paper and not cwd.is_dir():
        raise SystemExit(f"{paper} is not in release {version}; it first appears in "
                         f"{releases[paper]}.")
    if paper and used != releases[paper]:
        print(f"note: {paper} was published with emr-hyperneat {releases[paper]}; running it on "
              f"{used}. docs/reproducing-experiments.md says which releases give the same "
              f"results.", file=sys.stderr)
    # Resolve the program on the environment's PATH: on Windows a child process would
    # otherwise search the parent's PATH.
    program = shutil.which(command[0], path=env["PATH"]) or command[0]
    try:
        return subprocess.run([program, *command[1:]], cwd=cwd, env=env).returncode
    except FileNotFoundError:
        raise SystemExit(f"Command not found: {command[0]}")


def list_papers(releases: dict[str, str]) -> None:
    """Print each paper, its release, and whether its tag and checkout are here."""
    is_clone = _git("rev-parse", "--git-dir").returncode == 0
    print(f"{'paper':40} {'release':8} {'tag':10} checkout")
    for paper, version in sorted(releases.items()):
        tag = "-"
        if is_clone:
            have = _git("rev-parse", "--verify", "--quiet", f"refs/tags/v{version}").returncode == 0
            tag = f"v{version}" if have else "missing"
        ready = (checkout_dir(version) / ".venv" / _MARKER).exists()
        print(f"{paper:40} {version:8} {tag:10} {'ready' if ready else 'not set up'}")


def main(argv: Optional[list[str]] = None) -> int:
    """Parse the command line; everything after ``--`` is the command to run."""
    argv = sys.argv[1:] if argv is None else list(argv)
    command: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, command = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="each paper, its release, and whether it is set up")
    for name, text in (("setup", "build a release's checkout and environment"),
                       ("run", "run a command on a paper's release (after --)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("paper", nargs="?", help="a folder under papers/")
        p.add_argument("--release", help=f"a release (0.1.0 or v0.1.0), or {CURRENT}")
        p.add_argument("--python", default=sys.executable,
                       help="the Python that creates the environment (3.10 to 3.12)")
    args = ap.parse_args(argv)
    releases = load_releases()
    if args.action == "list":
        list_papers(releases)
        return 0
    version = resolve(args.paper, args.release, releases)
    if args.action == "setup":
        if version == CURRENT:
            raise SystemExit("--release current needs no setup: it is this checkout.")
        checkout = setup(version, args.python)
        folder = checkout / "papers" / args.paper if args.paper else checkout
        activate = venv_bin(checkout) / "activate"
        print(f"Ready. To work there:\n  cd {folder}\n  "
              f"{activate if os.name == 'nt' else f'source {activate}'}")
        return 0
    return run(args.paper, version, command, args.python, releases)


if __name__ == "__main__":
    sys.exit(main())
