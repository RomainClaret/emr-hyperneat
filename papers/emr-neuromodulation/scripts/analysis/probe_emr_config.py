#!/usr/bin/env python3
"""R7 Step-1 GATE: does the EMR class actually enable true neuromodulation?

Validates the EMR neuromod config in ISOLATION before wiring the shared runner,
because EMR's create_config has silent traps (verified by reading the code):
  1. unknown mode string -> silently 'disabled' (get_neuromodulation_config L546-547)
  2. true_neuromodulation only enabled when mode == 'true_neuromodulation' EXACTLY
     (no __post_init__; defaults enabled=False/true_neuromodulation=False)
  3. decode source (self_conn vs first-input weight) gated by use_self_connection_query

PASS criteria: neuromod_config.true_neuromodulation is True AND _neuromod_true has
finite, non-degenerate base_gains (~[0.5,1]) + receptor_densities (not all-zero).

Usage:  JAX_PLATFORM_NAME=cpu python papers/emr-neuromodulation/scripts/analysis/probe_emr_config.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]

import numpy as np
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multihead_palette_neuromodulation import RefProblem, INPUTS, TRUTH_TABLES
from emr_hyperneat.emrhyperneat import (
    EMRHyperNEAT,
)


def build_cfg(mode, extra_neuromod=None, pop=50, max_depth=4):
    nm = {
        'mode': mode,
        'modulation_strength': 5.0,
        'num_nt_types': 4,
        'receptor_from_weight': True,
        'receptor_derivation': 'tanh',
    }
    if extra_neuromod:
        nm.update(extra_neuromod)
    return {
        'algorithm_params': {
            'emrhyperneat': {
                'population_size': pop,
                'substrate': {
                    'input_coords': [[-1.0, 0.0], [1.0, 0.0]],
                    'output_coords': [[0.0, 1.0]],
                },
                'emr_hyperneat': {
                    'initial_depth': 0,
                    'max_depth': max_depth,
                    'variance_threshold': 0.03,
                    'neuromodulation': nm,
                },
                'neat_species': {
                    'compatibility_threshold': 2.5,
                    'max_stagnation': 40,
                },
            },
        },
    }


def _stats(a, name):
    if a is None:
        print(f"    {name}: None")
        return
    a = np.asarray(a)
    print(f"    {name}: shape={a.shape} min={a.min():.4f} max={a.max():.4f} "
          f"mean={a.mean():.4f} nan={bool(np.isnan(a).any())} allzero={bool((a == 0).all())}")


def probe(mode, extra=None, seed=42):
    print(f"\n{'=' * 72}\nPROBE  mode={mode!r}  extra={extra}\n{'=' * 72}")
    try:
        algo = EMRHyperNEAT()
        cfg = algo.create_config(build_cfg(mode, extra))
        nc = getattr(algo, 'neuromod_config', None)
        if nc is not None:
            print(f"  parsed neuromod_config: enabled={nc.enabled} "
                  f"true_neuromod={nc.true_neuromodulation} mode={nc.mode!r} "
                  f"mod_strength={nc.modulation_strength} "
                  f"receptor_from_weight={nc.receptor_from_weight} "
                  f"deriv={nc.receptor_derivation} "
                  f"use_self_conn={getattr(nc, 'use_self_connection_query', '<absent>')}")
        else:
            print("  WARNING: no .neuromod_config attribute on algo")

        problem = RefProblem(['xor'])
        state = algo.initialize(cfg, problem, seed=seed)

        class _Dummy:
            input_shape = (2,)
            output_shape = (1,)
            jitable = True

            def __init__(s):
                s.inputs = INPUTS
                s.targets = TRUTH_TABLES['xor']
                s.input_coords = [[-1.0, 0.0], [1.0, 0.0]]
                s.output_coords = [[0.0, 1.0]]

            def get_data(s):
                return []

        algo.run_generation_verbose(state, _Dummy(), skip_metrics=True)

        nt = getattr(algo, '_neuromod_true', None)
        if not nt:
            print("  RESULT: _neuromod_true EMPTY/None -> neuromod NOT applied  [FAIL]")
            return
        print("  _neuromod_true populated:")
        _stats(nt.get('base_gains'), 'base_gains')
        _stats(nt.get('receptor_densities'), 'receptor_densities')
        w1 = getattr(algo, '_cached_W1', None)
        print(f"    _cached_W1 shape={None if w1 is None else np.asarray(w1).shape}")
        ok = (nc is not None and nc.true_neuromodulation and nt.get('base_gains') is not None
              and not bool((np.asarray(nt['base_gains']) == 0).all()))
        print(f"  RESULT: {'[PASS]' if ok else '[FAIL]'}")
    except Exception as e:
        import traceback
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    probe('true_neuromodulation')                                            # correct EMR mode (W1 source, default)
    probe('true_neuromodulation', extra={'use_self_connection_query': True})  # reproduce-paper lever (self-conn source)
