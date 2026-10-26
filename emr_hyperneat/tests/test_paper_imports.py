"""Every paper script imports as `python <script>` would load it, without doing its work.

Two scripts of papers/emr-neuromodulation failed at import in 0.1.0: benchmark_seed_upgrade.py
imported the runner from its research-repository path, and probe_emr_config.py looked for it in its
own folder. For each paper, one subprocess from the paper's folder loads every script of its
scripts/ tree (the palette strategy modules aside, which the runners import) under a name other than
__main__, with the script's own folder first on the module path, as `python script.py` puts it
there. A script that does its work at module level is only compiled, and a script may exit when the
results it reads are missing. Between scripts the modules loaded from the paper folder are dropped,
so one script's local helper cannot stand in for another's.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Scripts with no __main__ guard, which run their analysis when loaded
COMPILE_ONLY = {
    'emr-dynamic-functions-bio-inspired': {
        'scripts/analysis/analysis_timescale_correlation.py',
        'scripts/analysis/verify_paper_numbers.py',
        'scripts/figures/generate_fig7_discovery_timing.py',
    },
}

DRIVER = r'''
import json
import os
import py_compile
import runpy
import sys
import warnings

warnings.simplefilter('ignore')
import jax, numpy, emr_hyperneat, tensorneat  # noqa: E401,F401  (kept loaded across scripts)
paper = os.getcwd()
scripts, compile_only = json.loads(sys.argv[1]), set(json.loads(sys.argv[2]))
base_path = list(sys.path)
found = {}
for rel in scripts:
    path = os.path.join(paper, rel)
    if rel in compile_only:
        py_compile.compile(path, doraise=True)
        found[rel] = 'compiled'
        continue
    sys.path[:] = [os.path.dirname(path)] + base_path
    try:
        runpy.run_path(path, run_name='__paper_import_check__')
        found[rel] = 'loaded'
    except SystemExit:
        found[rel] = 'exited'
    except BaseException as error:
        found[rel] = 'ERROR ' + type(error).__name__ + ': ' + str(error)[:400]
    for name, module in list(sys.modules.items()):
        if (getattr(module, '__file__', None) or '').startswith(paper + os.sep):
            del sys.modules[name]
print('IMPORTS', json.dumps(found))
'''


def _scripts(paper):
    folder = REPO / 'papers' / paper
    return sorted(str(p.relative_to(folder)) for p in folder.glob('scripts/**/*.py')
                  if '__pycache__' not in p.parts and 'palette_strategies' not in p.parts)


PAPERS = (sorted(p.name for p in (REPO / 'papers').iterdir() if (p / 'scripts').is_dir())
          if (REPO / 'papers').is_dir() else [])


@pytest.mark.skipif(not PAPERS, reason="needs the repository checkout")
@pytest.mark.parametrize('paper', PAPERS or ['none'])
def test_every_script_imports(paper):
    """No script of the paper fails at import (an exit for missing results is allowed)."""
    scripts = _scripts(paper)
    assert scripts
    env = {key: value for key, value in os.environ.items()
           if key not in ('JAX_PLATFORM_NAME', 'JAX_BACKEND_TARGET', 'PYTHONPATH')}
    env.update(JAX_PLATFORMS='cpu', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    compile_only = sorted(COMPILE_ONLY.get(paper, ()))
    result = subprocess.run(
        [sys.executable, '-c', DRIVER, json.dumps(scripts), json.dumps(compile_only)],
        cwd=REPO / 'papers' / paper, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith('IMPORTS ')][-1]
    found = json.loads(line[len('IMPORTS '):])
    failed = {script: outcome for script, outcome in found.items() if outcome.startswith('ERROR')}
    assert not failed, json.dumps(failed, indent=1)
    assert set(found) == set(scripts)
