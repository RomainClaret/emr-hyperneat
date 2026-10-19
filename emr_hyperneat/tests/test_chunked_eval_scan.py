"""The chunked h->h evaluation must be scan-based, not a launch loop.

A depth-5 probe measured the Python chunk loop at +116-126% per generation
(chunk 10 = 100 sequential launches). `chunked_vmap_eval` compiles the chunk
loop into one dispatch. The property pinned here is exactness on XLA:CPU and
agreement to float32 rounding on a GPU: chunked evaluation must equal the loop
form at the same chunk size, because each genome's evaluation is independent.
Against the unchunked vmap it is only close (see the test's comment). The
generation loop runs the same chunk loop through `chunked_forward_eval`, inside
a cached `jit`, so it compiles once per shape rather than every generation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conftest import (
    EMRHyperNEAT,
    GPU_TRAJECTORY_RTOL,
    XORProblem,
    assert_trajectories_agree,
    create_config_with_recurrence,
    exact_backend,
)
from emr_hyperneat.emrhyperneat import (
    _eval_single_fitness,
    _plain_recurrent_forward,
    chunked_forward_eval,
    chunked_vmap_eval,
)


class TestChunkedVmapEval:
    def test_equals_the_loop_form(self):
        # The helper's guarantee is exactness AGAINST THE LOOP FORM at the same
        # chunk size (same per-chunk computation, one dispatch). Chunked vs the
        # FULL vmap is only approximately equal: batched-matmul lowering can
        # differ by batch size (chunk=1 demonstrably does), which is float
        # noise, not a logic difference. On a GPU the loop-form guarantee is
        # also to float32 rounding: the lax.map body evaluates every chunk at
        # the full chunk shape (the last one padded) where the loop form's last
        # call has the remainder's shape, and the batched matmul is lowered per
        # shape (6e-8 relative at chunk 4 on an RTX 2080 Ti). XLA:CPU lowers
        # both identically, so CPU stays exact.
        key = jax.random.PRNGKey(0)
        W = jax.random.normal(key, (23, 4, 5))   # deliberately not a chunk multiple
        b = jax.random.normal(key, (23, 5))

        def fn(w, bias):
            return jnp.sum(jnp.tanh(w) @ bias[:, None])

        def loop_form(chunk):
            parts = []
            for start in range(0, W.shape[0], chunk):
                end = min(start + chunk, W.shape[0])
                parts.append(jax.vmap(fn)(W[start:end], b[start:end]))
            return jnp.concatenate(parts)

        full = jax.vmap(fn)(W, b)
        for chunk in (1, 4, 7, 23, 50):
            out = chunked_vmap_eval(fn, chunk, W, b)
            assert out.shape == (23,)
            loop = loop_form(min(chunk, 23))
            if exact_backend():
                assert bool(jnp.array_equal(out, loop)), (
                    f"chunk={chunk} diverged from the loop form")
            else:
                np.testing.assert_allclose(
                    np.asarray(out), np.asarray(loop), rtol=GPU_TRAJECTORY_RTOL,
                    atol=0.0, equal_nan=False,
                    err_msg=f"chunk={chunk} diverged from the loop form beyond "
                            f"float32 rounding")
            assert bool(jnp.allclose(out, full, atol=1e-6)), f"chunk={chunk} not close to full"

    def test_the_forward_form_equals_the_eager_form(self):
        """`chunked_forward_eval` runs `chunked_vmap_eval`'s chunk loop inside a `jax.jit` kept per
        forward and fitness mode, with the inputs and targets as its arguments, so a forward made
        once compiles it once per shape (`lax.map` builds its body on every call, so the eager
        loop compiled again every generation). The same values: bit for bit on CPU, to float32
        rounding on a GPU."""
        rng = np.random.default_rng(3)
        pop, n_in, n_pos, n_conn = 23, 3, 30, 50
        f32 = lambda a: jnp.asarray(a.astype(np.float32))  # noqa: E731
        arrays = (
            f32(rng.normal(size=(pop, n_in, n_pos))),
            f32(rng.normal(size=(pop, n_pos, 1))),
            jnp.asarray(rng.integers(0, n_pos, (pop, n_conn)).astype(np.int32)),
            jnp.asarray(rng.integers(0, n_pos, (pop, n_conn)).astype(np.int32)),
            f32(rng.normal(size=(pop, n_conn))),
            jnp.asarray(rng.uniform(size=(pop, n_conn)) < 0.8),
        )
        inputs = f32(rng.uniform(size=(4, n_in)))
        targets = f32(rng.uniform(size=(4, 1)))
        forward = _plain_recurrent_forward(5)
        assert _plain_recurrent_forward(5) is forward
        for mode in ("mse", "accuracy"):
            def eval_one(*args):
                return _eval_single_fitness(forward(inputs, *args), targets, mode)  # noqa: B023

            for chunk in (4, 7, 23):
                eager = chunked_vmap_eval(eval_one, chunk, *arrays)
                cached = chunked_forward_eval(forward, mode, inputs, targets, chunk, *arrays)
                assert cached.shape == (pop,)
                if exact_backend():
                    assert bool(jnp.array_equal(cached, eager)), f"{mode}, chunk {chunk}"
                else:
                    np.testing.assert_allclose(np.asarray(cached), np.asarray(eager),
                                               rtol=GPU_TRAJECTORY_RTOL, atol=0.0)

    def test_generation_with_eval_chunking_matches_unchunked(self):
        def run(chunk_size):
            algo = EMRHyperNEAT()
            config = create_config_with_recurrence("hidden_only")
            rec = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]["recurrence"]
            rec["hh_cache_enabled"] = False
            if chunk_size is not None:
                rec["eval_chunk_size"] = chunk_size
            cfg = algo.create_config(config)
            problem = XORProblem()
            state = algo.initialize(cfg, problem, seed=17)
            fits = []
            for _ in range(2):
                state, metrics = algo.run_generation(state, problem, verbose=False)
                fits.append(float(metrics.best_fitness))
            return fits

        # Bitwise on CPU; to float32 rounding on a GPU, where the batched evaluation is
        # lowered per batch size (chunk 7 against the whole population): jax 0.6.1 on an
        # RTX 2080 Ti happens to agree bitwise, jax 0.4.38 on the same card differs by 2.7e-7
        assert_trajectories_agree(
            {"chunk_7": run(chunk_size=7), "unchunked": run(chunk_size=None)},
            "eval chunking",
        )
