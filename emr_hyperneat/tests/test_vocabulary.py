"""`emr_hyperneat.vocabulary`: the engine's own objects, importable without TensorNEAT or the
configuration schemas."""
import subprocess
import sys

from emr_hyperneat import emrhyperneat as engine
from emr_hyperneat import vocabulary


def test_every_name_is_the_engines_own_object():
    assert len(vocabulary.__all__) == len(set(vocabulary.__all__)) == 23
    for name in vocabulary.__all__:
        assert getattr(vocabulary, name) is getattr(engine, name), name


def test_importing_it_loads_neither_tensorneat_nor_the_schemas():
    script = ("import sys, emr_hyperneat.vocabulary; "
              "print(sorted(m for m in ('pydantic', 'tensorneat') if m in sys.modules))")
    done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == '[]'
