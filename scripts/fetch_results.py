#!/usr/bin/env python3
"""Fetch the EMR-HyperNEAT experiment-result data release.

This repository ships code only. The experiment result data (the JSON outputs the
analysis and figure scripts read) is distributed as a separate archive on Zenodo.
This script resolves the **latest** version from the Zenodo concept DOI, downloads
that archive, verifies its checksum against what Zenodo publishes, and unpacks it
into each paper's ``results/`` (and ``data/``) directory in place. New data releases
(additional papers under the same code) are picked up automatically, with no edit to
this script.

Usage
-----
    python scripts/fetch_results.py                          # download latest + verify + unpack
    python scripts/fetch_results.py --archive <path.tar.gz>  # use a local copy (offline)
    python scripts/fetch_results.py --check                  # verify an already-downloaded archive
    python scripts/fetch_results.py --keep                   # keep the downloaded archive

After it runs, e.g.:
    papers/emr-dynamic-functions/results/...
    papers/emr-neuromodulation/results/...
    papers/emr-hyperneat/data/...
are populated, and the paper analysis/figure scripts run without re-running experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# --- Data release coordinates -------------------------------------------------
# The result data is archived on Zenodo. This is the *concept* DOI: it always
# resolves to the latest version, so new data releases are fetched automatically.
# The download URL and checksum for the current version are read from Zenodo at
# run time (see _resolve_latest), not hardcoded here.
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21383893"
_ZENODO_CONCEPT_RECORD = "21383893"
_ZENODO_API = f"https://zenodo.org/api/records/{_ZENODO_CONCEPT_RECORD}"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_latest() -> tuple[str, str, str, str]:
    """Query Zenodo for the latest version.

    Returns ``(download_url, md5, filename, version_doi)`` for the latest release's
    ``.tar.gz`` archive. Requesting the concept record returns the newest version.
    """
    with urllib.request.urlopen(_ZENODO_API) as resp:  # noqa: S310 (trusted Zenodo URL)
        meta = json.load(resp)
    archives = [f for f in meta.get("files", []) if f["key"].endswith(".tar.gz")]
    if not archives:
        raise SystemExit(f"No .tar.gz archive found on the Zenodo record ({_ZENODO_API}).")
    f = archives[0]
    return f["links"]["self"], f["checksum"].split(":", 1)[1], f["key"], meta.get("doi", "?")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected_md5: str) -> None:
    digest = _md5(path)
    if digest != expected_md5:
        raise SystemExit(
            f"Checksum mismatch for {path.name}:\n  expected md5 {expected_md5}\n  got md5      {digest}"
        )
    print(f"  checksum OK (md5 {digest[:16]}...)")


def _unpack(archive: Path) -> None:
    print(f"Unpacking into {REPO_ROOT}")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("papers/")]
        # guard against path traversal
        for m in members:
            target = (REPO_ROOT / m.name).resolve()
            if not str(target).startswith(str(REPO_ROOT)):
                raise SystemExit(f"Refusing to extract outside the repo: {m.name}")
        try:
            tar.extractall(REPO_ROOT, members=members, filter="data")
        except TypeError:  # filter= added in Python 3.12
            tar.extractall(REPO_ROOT, members=members)
    print(f"  unpacked {len(members)} entries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify a downloaded archive, do not unpack")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded archive")
    ap.add_argument("--archive", type=Path, default=None, help="path to a local archive (offline)")
    args = ap.parse_args()

    # Local archive: verify against the latest published checksum if Zenodo is reachable,
    # otherwise unpack it unverified (offline use).
    if args.archive:
        print(f"Using local archive {args.archive}")
        try:
            _, md5, _, doi = _resolve_latest()
            print(f"  latest release: DOI {doi}")
            _verify(args.archive, md5)
        except urllib.error.URLError:
            print("  warning: could not reach Zenodo for the checksum; unpacking unverified")
        _unpack(args.archive)
        return

    url, md5, name, doi = _resolve_latest()
    print(f"EMR-HyperNEAT data release (latest: {name}, DOI {doi}, concept {ZENODO_CONCEPT_DOI})")
    tmp = Path(tempfile.gettempdir()) / name

    if args.check:
        if not tmp.exists():
            raise SystemExit(f"No local archive at {tmp}; run without --check first.")
        _verify(tmp, md5)
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted Zenodo URL)
    print("Verifying checksum...")
    _verify(tmp, md5)
    _unpack(tmp)
    if not args.keep:
        tmp.unlink(missing_ok=True)
    print("Done. Each paper's results/ (and data/) are now populated.")


if __name__ == "__main__":
    sys.exit(main())
