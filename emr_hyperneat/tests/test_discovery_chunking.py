"""Bit-identity of population-chunked h->h discovery.

The discovery pipeline materializes population-dense (pop, A, A) tensors; the
recorded wall sits byte-exact at 1000x692^2x4 and 1000x888^2x4.
The fix chunks the population through the unchanged single-chunk pipeline with
the active-set union hoisted. The load-bearing property is NOT "it runs" but
"it computes the same thing": with the union pinned, a slice of the full call
and a call on the slice are the same computation, so forcing tiny chunks must
reproduce the unchunked run bit for bit -- fitness trajectory and connection
counts included.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_config_with_recurrence,
)


def _run(gens: int = 3, seed: int = 13):
    algo = EMRHyperNEAT()
    config = create_config_with_recurrence("hidden_only")
    # Fresh discovery every generation so every generation exercises the wrapper.
    config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"].setdefault(
        "recurrence", {})["hh_cache_enabled"] = False
    cfg = algo.create_config(config)
    problem = XORProblem()
    state = algo.initialize(cfg, problem, seed=seed)
    fits, conns = [], []
    for _ in range(gens):
        state, metrics = algo.run_generation(state, problem, verbose=False)
        fits.append(float(metrics.best_fitness))
        conns.append(int(getattr(algo, "_last_hh_connections", -1)))
    return fits, conns


class TestDiscoveryChunkingIdentity:
    def test_forced_chunking_is_bit_identical(self, monkeypatch):
        fits_ref, conns_ref = _run()

        # Force the wrapper into many small chunks regardless of fit.
        import emr_hyperneat.execution.chunking as chunking_mod
        monkeypatch.setattr(
            chunking_mod, "discovery_population_chunk", lambda pop, a, target_gb=2.0: 7)

        fits_chunked, conns_chunked = _run()
        assert fits_chunked == fits_ref, (fits_chunked, fits_ref)
        assert conns_chunked == conns_ref, (conns_chunked, conns_ref)

    def test_union_is_population_global_under_chunking(self, monkeypatch):
        """A per-chunk union would shrink A and change connection counts; the
        hoisted union must keep counts identical (asserted above) AND the
        wrapper must pass an explicit union into every chunk call."""
        import emr_hyperneat.emrhyperneat as emr_mod
        seen_unions = []
        original = emr_mod._discover_sparse_hh_multi_hop_single_chunk

        def spy(*args, **kwargs):
            seen_unions.append(kwargs.get("global_union_active"))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            emr_mod, "_discover_sparse_hh_multi_hop_single_chunk", spy)
        import emr_hyperneat.execution.chunking as chunking_mod
        monkeypatch.setattr(
            chunking_mod, "discovery_population_chunk", lambda pop, a, target_gb=2.0: 7)

        _run(gens=1)
        assert len(seen_unions) >= 2, "chunking did not engage"
        assert all(u is not None for u in seen_unions), (
            "a chunk call received union=None -- it would recompute a per-chunk union")
        first = seen_unions[0]
        assert all(bool(jnp.array_equal(u, first)) for u in seen_unions[1:])
