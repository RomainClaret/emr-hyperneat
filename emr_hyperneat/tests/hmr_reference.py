"""Recorded values of the frozen HMR engine, for the tests that compare EMR against it.

A test that compares EMR against HMR asks for HMR's side with
``hmr_value(case, compute, inputs=...)``: ``case`` names the place in the tests, ``compute`` would
return the value, and ``inputs`` are what it is computed from (arrays, configurations, problems,
seeds). In the research repository, where the frozen HMR engine lives, the same call runs HMR.
Here the call returns the value recorded for the case and for the sha256 of ``inputs`` in
hmr_reference_values.py, and never calls ``compute``: the tests of this package do not run HMR.

A recorded value is weaker than a live check. It holds for the inputs it was recorded with, on the
CPU architecture and the JAX version it was recorded with, and nothing here can tell whether HMR
itself would still compute it. So the call fails when the inputs no longer hash to a recorded value
(EMR's side of the comparison changed, or the test did), and skips where no value applies: off the
CPU, on another JAX version, or on an architecture with no record.

The encoding and the hash are the research module's, function by function (its recorder refuses to
write otherwise): arrays by dtype, shape and bytes (zlib-compressed, in base64), floats by
``float.hex``, a problem by its class name, its data and its other plain attributes.
"""
import base64
import hashlib
import json
import os
import platform
import zlib
from typing import Any, Callable

import numpy as np
import pytest


def platform_key() -> str:
    """The CPU architecture a value was computed on: 'arm64' or 'x86_64'."""
    machine = platform.machine().lower()
    return {'aarch64': 'arm64', 'amd64': 'x86_64'}.get(machine, machine)


def encode(value: Any) -> Any:
    """A value as an exact literal of Python's basic types: arrays by dtype, shape and
    zlib-compressed bytes (base64), floats by ``float.hex``."""
    return _literal(value, compress=True)


def _literal(value: Any, compress: bool) -> Any:
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, str):
        return bool(value) if isinstance(value, np.bool_) else value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.ndarray):
        return int(value)
    if isinstance(value, float):
        return {'float': float(value).hex()}
    if isinstance(value, np.floating):
        array = np.asarray(value)
        return {'scalar': array.dtype.str, 'hex': array.tobytes().hex()}
    if isinstance(value, tuple):
        return {'tuple': [_literal(item, compress) for item in value]}
    if isinstance(value, list):
        return [_literal(item, compress) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_literal(item, compress) for item in value]
        return {'set': sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, dict):
        return {'dict': [[_literal(k, compress), _literal(v, compress)]
                         for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]}
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f'cannot encode {type(value).__name__}')
    data = np.ascontiguousarray(array).tobytes()
    if compress:
        return {'array': array.dtype.str, 'shape': list(array.shape),
                'zlib64': base64.b64encode(zlib.compress(data, 9)).decode('ascii')}
    return {'array': array.dtype.str, 'shape': list(array.shape), 'hex': data.hex()}


def decode(literal: Any) -> Any:
    """The value of an encoded literal (the inverse of :func:`encode`)."""
    if isinstance(literal, list):
        return [decode(item) for item in literal]
    if not isinstance(literal, dict):
        return literal
    if 'float' in literal:
        return float.fromhex(literal['float'])
    if 'scalar' in literal:
        return np.frombuffer(bytes.fromhex(literal['hex']), dtype=literal['scalar'])[0]
    if 'array' in literal:
        data = zlib.decompress(base64.b64decode(literal['zlib64'])) if 'zlib64' in literal else \
            bytes.fromhex(literal['hex'])
        return np.frombuffer(data, dtype=literal['array']).reshape(literal['shape']).copy()
    if 'tuple' in literal:
        return tuple(decode(item) for item in literal['tuple'])
    if 'set' in literal:
        return {decode(item) for item in literal['set']}
    if 'dict' in literal:
        return {_hashable(decode(k)): decode(v) for k, v in literal['dict']}
    raise ValueError(f'not an encoded value: {literal!r}')


def _hashable(key: Any) -> Any:
    return tuple(key) if isinstance(key, list) else key


def digest(inputs: Any) -> str:
    """The sha256 of a test's inputs, or of a value: arrays by dtype, shape and bytes, everything
    else by its literal."""
    return hashlib.sha256(json.dumps(_literal(_as_plain(inputs), compress=False),
                                     sort_keys=True).encode()).hexdigest()


def _as_plain(value: Any) -> Any:
    """jax arrays as numpy arrays and problems as their data, recursively."""
    if isinstance(value, (list, tuple)):
        return type(value)(_as_plain(item) for item in value)
    if isinstance(value, dict):
        return {k: _as_plain(v) for k, v in value.items()}
    if hasattr(value, '__array__') and not isinstance(value, np.ndarray):
        return np.asarray(value)
    if callable(getattr(value, 'get_data', None)):
        return _problem(value)
    return value


def _problem(problem: Any) -> dict:
    """A problem as what an engine reads from it: its class name, its data and its other plain
    attributes."""
    plain = (bool, int, float, str, list, tuple, type(None))
    attributes = {name: getattr(problem, name) for name in dir(problem) if not name.startswith('_')}
    return {'problem': type(problem).__name__, 'get_data': _as_plain(problem.get_data()),
            **{name: _as_plain(value) for name, value in attributes.items()
               if isinstance(value, plain)}}


def live(name: str, *args: Any, **kwargs: Any) -> Any:
    """The research repository's live computation of HMR's value; the tests here do not run HMR."""
    raise RuntimeError(f'live({name!r}): the tests of this package do not run HMR; hmr_value '
                       f'returns the recorded value')


def hmr_value(case: str, compute: Callable[[], Any], *, inputs: Any = ()) -> Any:
    """HMR's side of a comparison: the value recorded for ``case`` and these inputs.

    ``compute`` is research's live computation and is never called here.
    """
    import jax

    from hmr_reference_values import RECORDED  # type: ignore[import-not-found]

    if case not in RECORDED:
        pytest.fail(f'no HMR value is recorded for {case!r}; record it with the research '
                    f'repository\'s scripts/release/record_hmr_reference.py')
    platforms = RECORDED[case]['platforms']
    machine = platform_key()
    if machine not in platforms:
        pytest.skip(f'{case}: HMR values are recorded on {", ".join(platforms)}, not on {machine}')
    record = platforms[machine]
    if jax.__version__ != record['jax']:
        pytest.skip(f'{case}: HMR values are recorded with jax {record["jax"]}, this is '
                    f'{jax.__version__}')
    if jax.default_backend() != 'cpu':
        pytest.skip(f'{case}: HMR values are recorded on the CPU; EMR on the '
                    f'{jax.default_backend()} backend rounds otherwise')
    sha = digest(inputs)
    if sha not in record['values']:
        flags = os.environ.get('XLA_FLAGS', '')
        # The XLA_FLAGS conditions the values were recorded under: a list, or one string in a file
        # recorded before conditions were merged
        recorded = record['xla_flags']
        conditions = [recorded] if isinstance(recorded, str) else list(recorded)
        hint = '' if flags in conditions else (
            f' (XLA_FLAGS here {flags!r}, the values were recorded under {conditions!r})')
        pytest.fail(f'{case}: the inputs (sha256 {sha}) are not the ones HMR values were recorded '
                    f'for ({", ".join(sorted(record["values"]))}): EMR\'s side of the comparison '
                    f'changed, or the test did, so no recorded value applies; record again{hint}')
    return decode(record['values'][sha]['value'])
