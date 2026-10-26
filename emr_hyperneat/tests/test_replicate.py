"""Tests for ``scripts/replicate.py``, which runs a paper on the release it names.

The release map and the docs that repeat it are checked against the repository. The git steps run
in a temporary repository with a tag, isolated from the user's git configuration, and the
environment build is replaced by a stub, so the tests need no network and install nothing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import emr_hyperneat

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "replicate.py"
_PAPERS = sorted(json.loads((REPO / "papers" / "releases.json").read_text()))


def _load(path: Path):
    """Load a copy of the script; its repository root is two levels above ``path``."""
    spec = importlib.util.spec_from_file_location(f"replicate_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load(_SCRIPT)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


# The release map, and the docs that repeat it

def test_every_paper_folder_names_a_release():
    folders = sorted(p.name for p in (REPO / "papers").iterdir() if p.is_dir())
    assert sorted(rp.load_releases()) == folders


def test_releases_are_versions_up_to_this_one():
    for paper, version in rp.load_releases().items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), paper
        assert _version(version) <= _version(emr_hyperneat.__version__), paper


@pytest.mark.parametrize("paper", _PAPERS)
def test_readmes_name_the_same_release(paper):
    version = rp.load_releases()[paper]
    header = (REPO / "papers" / paper / "README.md").read_text()
    assert re.search(rf"Code release: \**emr-hyperneat {re.escape(version)}\**", header), paper
    assert f"tree/v{version}/papers/{paper}" in (REPO / "README.md").read_text(), paper


def test_list_names_every_paper_and_release(capsys):
    rp.list_papers(rp.load_releases())
    out = capsys.readouterr().out
    for paper, version in rp.load_releases().items():
        assert re.search(rf"^{re.escape(paper)}\s+{re.escape(version)}\s", out, re.M), paper


# Choosing a release

def test_normalize_accepts_versions_tags_and_current():
    assert rp.normalize("0.1.0") == "0.1.0"
    assert rp.normalize("v0.1.0") == "0.1.0"
    assert rp.normalize("current") == "current"
    with pytest.raises(SystemExit, match="Not a release"):
        rp.normalize("main")


def test_resolve_defaults_to_the_papers_release():
    releases = {"paper-a": "0.1.0"}
    assert rp.resolve("paper-a", None, releases) == "0.1.0"
    assert rp.resolve("paper-a", "v0.2.0", releases) == "0.2.0"
    assert rp.resolve(None, "current", releases) == "current"
    with pytest.raises(SystemExit, match="Unknown paper"):
        rp.resolve("paper-b", None, releases)
    with pytest.raises(SystemExit, match="Name a paper"):
        rp.resolve(None, None, releases)


# Building a release checkout, in a temporary repository

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def tagged_repo(tmp_path, monkeypatch):
    """A repository whose tag v0.0.1 has ``papers/demo``, with one later commit on top."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)  # no signing, hooks or aliases
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.org")
    repo = tmp_path / "repo"
    (repo / "papers" / "demo").mkdir(parents=True)
    (repo / "papers" / "demo" / "README.md").write_text("demo\n")
    (repo / "papers" / "releases.json").write_text(json.dumps({"demo": "0.0.1"}))
    (repo / "emr_hyperneat").mkdir()
    (repo / "emr_hyperneat" / "_version.py").write_text('__version__ = "0.0.2"\n')
    (repo / "scripts").mkdir()
    (repo / "scripts" / "replicate.py").write_text(_SCRIPT.read_text())
    (repo / ".gitignore").write_text(".replicate/\n__pycache__/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "release")
    _git(repo, "tag", "v0.0.1")
    (repo / "later.txt").write_text("after the release\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "later")
    return repo


def test_setup_checks_out_the_tag_once(tagged_repo):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    builds = []
    checkout = script.setup("0.0.1", make_env=lambda c, py: builds.append(c), check=False)
    assert checkout == tagged_repo / ".replicate" / "v0.0.1"
    assert _git(checkout, "rev-parse", "HEAD") == _git(tagged_repo, "rev-parse", "v0.0.1^{commit}")
    assert (checkout / "papers" / "demo" / "README.md").exists()
    assert not (checkout / "later.txt").exists()
    assert script.setup("0.0.1", make_env=lambda c, py: builds.append(c), check=False) == checkout
    assert builds == [checkout]  # the second call finds the environment built
    assert ".replicate" not in _git(tagged_repo, "status", "--porcelain")  # ignored, as here


def test_setup_recovers_a_checkout_deleted_by_hand(tagged_repo):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    checkout = script.setup("0.0.1", make_env=lambda c, py: None, check=False)
    shutil.rmtree(checkout)  # git still lists the worktree until it is pruned
    assert script.setup("0.0.1", make_env=lambda c, py: None, check=False) == checkout
    assert (checkout / "papers" / "demo" / "README.md").exists()


def test_a_missing_tag_says_how_to_fetch_it(tagged_repo):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    with pytest.raises(SystemExit, match="git fetch --tags origin"):
        script.setup("9.9.9", make_env=lambda c, py: None, check=False)


def test_outside_a_clone_says_how_to_clone(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    copy = tmp_path / "download" / "scripts" / "replicate.py"
    copy.parent.mkdir(parents=True)
    copy.write_text(_SCRIPT.read_text())
    with pytest.raises(SystemExit, match="git clone --recursive"):
        _load(copy).tag_commit("0.1.0")


# Running a command

@pytest.fixture
def recorded_run(monkeypatch):
    """Replace subprocess.run with a recorder; return the dict it fills."""
    calls: dict = {}

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        calls.update(cmd=cmd, cwd=cwd, env=env)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_run_uses_the_release_environment_in_the_paper_folder(tagged_repo, monkeypatch,
                                                               recorded_run):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    checkout = tagged_repo / ".replicate" / "v0.0.1"
    (checkout / "papers" / "demo").mkdir(parents=True)
    monkeypatch.setattr(script, "setup", lambda version, python: checkout)
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert script.run("demo", "0.0.1", ["python", "-c", "pass"], releases={"demo": "0.0.1"}) == 0
    assert recorded_run["cwd"] == checkout / "papers" / "demo"
    assert recorded_run["cmd"][1:] == ["-c", "pass"]
    env = recorded_run["env"]
    assert "PYTHONPATH" not in env
    assert env["VIRTUAL_ENV"] == str(checkout / ".venv")
    assert env["PATH"].split(os.pathsep)[0] == str(script.venv_bin(checkout))


def test_run_without_a_paper_runs_at_the_checkout_root(tagged_repo, monkeypatch, recorded_run):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    checkout = tagged_repo / ".replicate" / "v0.0.1"
    monkeypatch.setattr(script, "setup", lambda version, python: checkout)
    script.run(None, "0.0.1", ["python", "scripts/fetch_results.py"], releases={"demo": "0.0.1"})
    assert recorded_run["cwd"] == checkout


def test_run_current_stays_here_and_notes_the_release(tagged_repo, monkeypatch, recorded_run,
                                                      capsys):
    script = _load(tagged_repo / "scripts" / "replicate.py")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    script.run("demo", "current", ["python", "-c", "pass"], releases={"demo": "0.0.1"})
    assert recorded_run["cwd"] == tagged_repo / "papers" / "demo"
    assert recorded_run["env"]["PYTHONPATH"] == "/elsewhere"
    # `python` in the command is the interpreter that runs the script
    assert recorded_run["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert "published with emr-hyperneat 0.0.1; running it on 0.0.2" in capsys.readouterr().err


def test_main_splits_the_command_at_the_double_dash(monkeypatch):
    seen = {}
    monkeypatch.setattr(rp, "run", lambda paper, version, command, python, releases:
                        seen.update(paper=paper, version=version, command=command) or 0)
    paper = _PAPERS[0]
    assert rp.main(["run", paper, "--release", "v0.1.0", "--", "python", "-m", "pytest", "-q"]) == 0
    assert seen == {"paper": paper, "version": "0.1.0", "command": ["python", "-m", "pytest", "-q"]}
