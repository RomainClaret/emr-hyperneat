"""The engine's layout: `emr_hyperneat.emrhyperneat` re-exports what lives in `emr_hyperneat.engine`.

The engine's code lives in the modules of `emr_hyperneat.engine`, and `emr_hyperneat.emrhyperneat` imports every
name back, so each keeps the import path it had when the whole engine was one file. These tests hold that
contract: the names that paper scripts, emr-picbreeder and research harnesses import are there and are the
engine's own objects, and a name that a test or a harness replaces on `emr_hyperneat.emrhyperneat` reaches the
code that reads it, wherever that code lives.
"""
import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import emr_hyperneat.emrhyperneat as facade
import emr_hyperneat.engine as engine_package

ENGINE_DIR = Path(engine_package.__file__).parent

# Names that tests and research harnesses replace on emr_hyperneat.emrhyperneat (monkeypatch or assignment)
PATCHED = (
    'discover_sparse_hh_vectorized_multi_hop', '_discover_sparse_hh_multi_hop_single_chunk',
    '_per_genome_block_for', '_PER_GENOME_BLOCK_PAIRS', 'aggregation_from_weight_interpretation',
    'leo_sparsify', 'modulatory_neuron_masks', 'sanitize_fitness', 'split_ranges',
    'evaluate_population_on_device', 'evaluation_ranges', 'query_population_split',
    'evaluate_population_split', '_pmap_data_parallel_full_hh', 'strong_default_float',
    'sparse_cap_refusal',
)

# Names that code outside the package imports from emr_hyperneat.emrhyperneat: the paper scripts, emr-picbreeder,
# the digest harness and research harnesses
EXTERNAL = (
    'EMRHyperNEAT', 'ACTIVATION_LIST', 'ACTIVATION_FUNCTIONS', 'AGGREGATION_LIST', 'derive_receptor_from_weight',
    'NT_TASK_PRESETS', 'continuous_to_index', 'cppn_output_function_indices', 'substrate_weight_matrices',
    'self_connection_rows', 'WEIGHT_THRESHOLD', 'sparse_cap_refusal', '_last_discovery_stats',
    '_pmap_data_parallel_full_hh',
)


def engine_modules() -> list:
    return [importlib.import_module(info.name)
            for info in pkgutil.iter_modules(engine_package.__path__, 'emr_hyperneat.engine.')]


def defined_names(module) -> set:
    """Names a module defines at its top level (its imports excluded)."""
    tree = ast.parse(Path(module.__file__).read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def test_external_names_import_from_the_facade():
    missing = [name for name in EXTERNAL if not hasattr(facade, name)]
    assert not missing


def test_the_facade_holds_the_engine_objects():
    # One object per name: a registry, a cache or a class is the engine's own, not a copy
    for module in engine_modules():
        for name in defined_names(module):
            assert facade.__dict__.get(name) is module.__dict__[name], f'{module.__name__}.{name}'


@pytest.mark.parametrize('name', PATCHED)
def test_a_patch_on_the_facade_reaches_the_engine(monkeypatch, name):
    original = getattr(facade, name)
    holders = [m for m in engine_modules() if m.__dict__.get(name) is original]
    homes = [m for m in engine_modules() if name in defined_names(m)]
    assert all(m in holders for m in homes)
    sentinel = object()
    monkeypatch.setattr(facade, name, sentinel)
    assert all(m.__dict__[name] is sentinel for m in holders)
    monkeypatch.undo()
    assert getattr(facade, name) is original
    assert all(m.__dict__[name] is original for m in holders)


def test_engine_modules_bind_package_names_under_their_own_names():
    # The facade forwards a patch by name: an alias would hold the object under a name no patch sets
    aliased = []
    for path in sorted(ENGINE_DIR.glob('*.py')):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or '').startswith('emr_hyperneat')):
                aliased += [f'{path.name}: {a.name} as {a.asname}' for a in node.names
                            if a.asname and a.asname != a.name]
    assert not aliased


def test_engine_modules_do_not_import_the_facade_at_import_time():
    # The facade imports the engine modules first; one importing it back would see it half built
    offenders = []
    for path in sorted(ENGINE_DIR.glob('*.py')):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.module == 'emr_hyperneat.emrhyperneat':
                offenders.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module == 'emr_hyperneat' and \
                    any(a.name == 'emrhyperneat' for a in node.names):
                offenders.append(path.name)
            elif isinstance(node, ast.Import) and any(a.name == 'emr_hyperneat.emrhyperneat' for a in node.names):
                offenders.append(path.name)
    assert not offenders
