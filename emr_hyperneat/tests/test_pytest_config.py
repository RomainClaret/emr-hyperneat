"""The root pytest.ini repeats emr_hyperneat/tests/pytest.ini below its discovery lines.

A bare `pytest` from the repository root reads the root file (0.1.0 had none, and such a run
collected the TensorNEAT submodule's own tests and broke every later `import tensorneat`). The
documented form, `pytest emr_hyperneat/tests ...`, reads the tests' file. If the two drifted, a
bare run would register other markers or options.

The nightly workflow runs the test files its SHARDS list assigns and stops at once when a file is
unassigned or assigned twice, so the assignment is checked here too, on every run of the suite.
"""

import configparser
import pathlib

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]


def _options(path):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    return dict(parser['pytest'])


@pytest.mark.skipif(not (REPO / 'pytest.ini').is_file(), reason="needs the repository checkout")
def test_root_ini_repeats_the_tests_ini():
    root = _options(REPO / 'pytest.ini')
    tests = _options(REPO / 'emr_hyperneat' / 'tests' / 'pytest.ini')
    assert root.pop('testpaths') == 'emr_hyperneat/tests'
    assert 'third_party' in root.pop('norecursedirs').split()
    assert tests.pop('testpaths') == '.'
    assert root == tests


@pytest.mark.skipif(not (REPO / '.github' / 'workflows' / 'ci.yml').is_file(),
                    reason="needs the repository checkout")
def test_every_test_file_has_one_nightly_shard():
    """Read as the workflow runs it: each SHARDS line starts with a label of the matrix, so a file on
    a line no shard selects (a wrapped line, a label outside the matrix) fails here."""
    workflow = yaml.safe_load((REPO / '.github' / 'workflows' / 'ci.yml').read_text())
    labels = [str(label) for label in workflow['jobs']['full-suite']['strategy']['matrix']['shard']]
    assigned = {}
    for line in workflow['env']['SHARDS'].splitlines():
        words = line.split()
        if not words:
            continue
        assert words[0] in labels, f"a SHARDS line starts with {words[0]!r}, not a matrix label"
        assert all(word.endswith('.py') for word in words[1:]), line
        assigned.setdefault(words[0], []).extend(words[1:])
    assert all(assigned.get(label) for label in labels), "a shard of the matrix has no test file"
    run = [name for label in labels for name in assigned[label]]
    on_disk = sorted(path.name for path in (REPO / 'emr_hyperneat' / 'tests').glob('test_*.py'))
    assert len(run) == len(set(run)), "a test file is in two shards"
    assert sorted(run) == on_disk
