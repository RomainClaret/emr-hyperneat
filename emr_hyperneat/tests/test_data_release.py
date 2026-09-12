"""Tests for the Zenodo data-release fetch (``scripts/fetch_results.py``).

Two layers:

1. Offline / hermetic tests that exercise the script's parse, verify, and unpack
   logic with a mocked Zenodo response and small in-memory archives. These run
   everywhere, including CI, with no network.
2. Live ``network``-marked tests that confirm the real Zenodo record resolves and
   serves the archive. They auto-skip when Zenodo is unreachable, so an offline
   machine or a transient outage skips rather than fails.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# fetch_results.py lives in scripts/, which is excluded from the installed package,
# so load it directly by path.
_FR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fetch_results.py"
_spec = importlib.util.spec_from_file_location("fetch_results", _FR_PATH)
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)


# --- Offline: parsing the Zenodo response -------------------------------------

_FAKE_ZENODO = {
    "doi": "10.5281/zenodo.21383894",
    "files": [
        {"key": "README.md", "checksum": "md5:" + "0" * 32, "links": {"self": "https://x/readme"}},
        {
            "key": "emr-hyperneat-data-v0.1.0.tar.gz",
            "checksum": "md5:f019e9691356535954ccb09919e1be9a",
            "links": {
                "self": "https://zenodo.org/api/records/21383894/files/"
                "emr-hyperneat-data-v0.1.0.tar.gz/content"
            },
        },
    ],
}


def test_resolve_latest_parses_zenodo_json(monkeypatch):
    """_resolve_latest picks the .tar.gz and returns (url, md5, name, doi)."""
    payload = json.dumps(_FAKE_ZENODO).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))

    url, md5, name, doi = fr._resolve_latest()

    assert name == "emr-hyperneat-data-v0.1.0.tar.gz"  # the .tar.gz, not README.md
    assert md5 == "f019e9691356535954ccb09919e1be9a"  # the "md5:" prefix is stripped
    assert url.endswith(".tar.gz/content")
    assert doi == "10.5281/zenodo.21383894"


def test_resolve_latest_errors_without_archive(monkeypatch):
    """A record with no .tar.gz is a clear failure, not a silent wrong result."""
    payload = json.dumps({"doi": "x", "files": [{"key": "notes.txt", "checksum": "md5:0",
                                                 "links": {"self": "u"}}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(SystemExit):
        fr._resolve_latest()


# --- Offline: checksum verification -------------------------------------------

def test_verify_accepts_and_rejects_md5(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello data release")
    good = hashlib.md5(f.read_bytes()).hexdigest()

    fr._verify(f, good)  # matching md5: no error
    with pytest.raises(SystemExit):
        fr._verify(f, "0" * 32)  # wrong md5


# --- Offline: unpack only papers/ + block path traversal ----------------------

def _make_targz(dest: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(dest, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return dest


def test_unpack_extracts_only_papers_members(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(fr, "REPO_ROOT", root)
    archive = _make_targz(
        tmp_path / "a.tar.gz",
        {"papers/emr-x/results/r.json": b"{}", "outside.txt": b"nope"},
    )

    fr._unpack(archive)

    assert (root / "papers/emr-x/results/r.json").exists()
    assert not (root / "outside.txt").exists()  # non-papers/ members are skipped


def test_unpack_blocks_path_traversal(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(fr, "REPO_ROOT", root)
    archive = _make_targz(tmp_path / "evil.tar.gz", {"papers/../../evil.json": b"x"})

    with pytest.raises(SystemExit):  # target resolves outside REPO_ROOT
        fr._unpack(archive)
    assert not (tmp_path / "evil.json").exists()


# --- Live: Zenodo is actually reachable and serving the archive ---------------

@pytest.mark.network
def test_zenodo_live_resolves_and_reachable():
    """The real concept record resolves to a well-formed, downloadable archive."""
    try:
        url, md5, name, doi = fr._resolve_latest()
    except OSError as e:  # URLError / timeout / DNS: offline, not a regression
        pytest.skip(f"Zenodo unreachable: {e}")

    assert "zenodo.org" in url
    assert len(md5) == 32 and all(c in "0123456789abcdef" for c in md5)
    assert name.endswith(".tar.gz")
    assert doi.startswith("10.5281/zenodo.")

    try:  # confirm Zenodo serves the file, without pulling the whole ~9.5 MB
        with urllib.request.urlopen(url, timeout=30) as resp:
            head = resp.read(8192)
    except OSError as e:
        pytest.skip(f"Zenodo archive unreachable: {e}")
    assert head, "archive URL returned no data"


@pytest.mark.network
@pytest.mark.slow
def test_zenodo_archive_downloads_and_md5_matches(tmp_path):
    """End-to-end: the full archive downloads and its md5 matches Zenodo's."""
    try:
        url, md5, name, _ = fr._resolve_latest()
        dest = tmp_path / name
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted Zenodo URL)
    except OSError as e:
        pytest.skip(f"Zenodo unreachable: {e}")
    fr._verify(dest, md5)  # raises SystemExit on mismatch
