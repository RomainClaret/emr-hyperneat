"""The root pytest.ini repeats emr_hyperneat/tests/pytest.ini below its discovery lines.

A bare `pytest` from the repository root reads the root file (0.1.0 had none, and such a run
collected the TensorNEAT submodule's own tests and broke every later `import tensorneat`). The
documented form, `pytest emr_hyperneat/tests ...`, reads the tests' file. If the two drifted, a
bare run would register other markers or options.
"""

import configparser
import pathlib

import pytest

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
