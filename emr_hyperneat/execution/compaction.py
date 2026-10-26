"""Active-set compaction for the recurrent forward pass.

The recurrent path builds a dense hidden-to-hidden matrix over every candidate
position:

    Whh = jnp.zeros((num_positions, num_positions)).at[from, to].add(weights)

That is `O(N^2)` in the substrate ceiling, and the ceiling is enormous while the
realized substrate is tiny. At depth 7, `N = 87,380`, so `Whh` alone is 30.5 GB per
genome; the measured realized substrate on XOR is about 4-7 positions. HMR measured
~51 active of 5,592,404 at depth 10, i.e. the dense buffers were ~10^5x larger than
the data they carried.

The feedforward path escapes this because it chunks positions, and its chunks are
independent. The recurrent path cannot: the h->h scan gathers and scatters across
arbitrary positions, so no chunk is self-contained. It needs compaction instead.

The dense `Whh` stays, because a scatter-add in its place would be the CPU bottleneck
that makes recurrent topologies look infeasible. Compaction shrinks the *space* it
is built over: evaluate on the union of positions any genome actually reached, and
the matmul stays dense and fast while the matrix becomes negligible.

The remap has to be **exact**, not approximate: a wrong index would silently fold
every h->h edge onto one node and still produce plausible-looking numbers rather
than raising. `active` is sorted, so `searchsorted` recovers the compact index of a
genuinely active position exactly. Padded and invalid entries may land anywhere;
they are neutralized downstream by `valid_mask` (`effective_weights =
where(valid_mask, w, 0)`) and clipped into range here.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["ActiveSet", "find_active_positions", "compact_weights", "remap_sparse_hh",
           "bucket_size", "bucket_active_set", "pad_positions"]


class ActiveSet(NamedTuple):
    """Positions any genome in the population reached, and their count.

    Attributes:
        indices: Sorted global position indices, shape (num_active,).
        num_active: Number of active positions; 0 means no genome reached anything.
        total_positions: The uncompacted ceiling, kept for reporting the reduction.
    """

    indices: np.ndarray
    num_active: int
    total_positions: int
    size: Optional[int] = None
    """Length of the compact arrays when they are padded (`bucket_active_set`); None means
    exactly `num_active`. Positions `num_active .. size - 1` are virtual: zero weights, no
    connection, mask False."""

    @property
    def length(self) -> int:
        """Length of the compact position axis the evaluation sees."""
        return self.num_active if self.size is None else self.size

    @property
    def search_indices(self) -> np.ndarray:
        """Sorted indices over the whole compact axis: the real positions, then virtual ones.

        The virtual positions are numbered from `total_positions` up, so they sort after every
        real position and `searchsorted` still returns the exact compact index of a real one.
        """
        pad = self.length - self.num_active
        if pad <= 0:
            return self.indices
        virtual = np.arange(self.total_positions, self.total_positions + pad,
                            dtype=self.indices.dtype)
        return np.concatenate([self.indices, virtual])

    @property
    def reduction(self) -> float:
        """How many times smaller the compact space is. 1.0 when nothing is saved."""
        if self.num_active == 0:
            return float(self.total_positions)
        return self.total_positions / self.num_active


def find_active_positions(masks: Any) -> ActiveSet:
    """Union of positions reached by any genome.

    Computed on the host deliberately. The population mask is `(pop, N)` and the
    result is a list of indices; doing this on the device would allocate an `O(N)`
    index array there, which is the allocation being avoided.

    Args:
        masks: Boolean-ish array of shape (pop_size, total_positions).

    Returns:
        The active set. `indices` is sorted ascending, which `remap_sparse_hh`
        relies on.
    """
    masks_np = np.asarray(masks)
    if masks_np.ndim != 2:
        raise ValueError(
            f"masks must be (pop_size, total_positions), got shape {masks_np.shape}"
        )
    total_positions = int(masks_np.shape[1])
    indices = np.nonzero(np.any(masks_np.astype(bool), axis=0))[0]
    return ActiveSet(
        indices=indices,
        num_active=int(indices.size),
        total_positions=total_positions,
    )


def bucket_size(num_active: int, total_positions: int) -> int:
    """The padded length for `num_active` real positions.

    Rounded up to a step of `max(64, 2 ** (floor(log2 n) - 3))`: at most 63 positions of
    padding below 512, at most an eighth of `n` above, about eight sizes per doubling, and
    never beyond the uncompacted ceiling (nor below `n`).
    """
    n = int(num_active)
    if n <= 0:
        return 0
    step = max(64, 1 << max(0, n.bit_length() - 1 - 3))
    size = -(-n // step) * step
    return max(n, min(size, int(total_positions)))


def bucket_active_set(active: ActiveSet) -> ActiveSet:
    """The same active set with its compact axis padded to `bucket_size`."""
    return active._replace(size=bucket_size(active.num_active, active.total_positions))


def pad_positions(values: Any, active: ActiveSet, axis: int, fill: Any = 0) -> jnp.ndarray:
    """Gather the real positions of `values` along `axis` on the host and pad with `fill`.

    The gather runs in numpy: done on the device, a gather to a new length would compile a new
    executable, which is what the padding is there to avoid.
    """
    host = np.asarray(values)
    shape = list(host.shape)
    shape[axis] = active.length
    out = np.full(shape, fill, dtype=host.dtype)
    index = [slice(None)] * host.ndim
    index[axis] = slice(0, active.num_active)
    out[tuple(index)] = np.take(host, active.indices, axis=axis)
    return jax.device_put(out)


def compact_weights(
    W1: Any,
    W2: Any,
    active: ActiveSet,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Restrict the weight matrices to the active positions.

    Gathering happens on the host before the device transfer, so the full matrices
    never need to be resident on the device. For a memmap-backed array this touches
    only the pages holding the selected columns.

    **Both inputs carry positions on the last axis.** That is the host buffer layout
    the offload path builds (`W2_cpu = np.zeros((pop, num_outputs, total_positions))`),
    while the *evaluator* consumes W2 transposed, as `(pop, positions, n_out)`.
    Gathering on the wrong axis here would silently index the output dimension
    instead of the position dimension.

    Args:
        W1: Input-to-hidden weights, (pop, n_in, total_positions).
        W2: Hidden-to-output weights, (pop, n_out, total_positions): the host layout,
            not the transposed evaluation layout.
        active: Active set from `find_active_positions`.

    Returns:
        Tuple of (W1_active, W2_active) on the device, with the last axis reduced
        to `active.num_active`.

    Raises:
        ValueError: If either input does not carry positions on the last axis.
    """
    W1_np, W2_np = np.asarray(W1), np.asarray(W2)
    for name, array in (("W1", W1_np), ("W2", W2_np)):
        if array.ndim != 3 or array.shape[2] != active.total_positions:
            raise ValueError(
                f"{name} must be 3-D with positions last "
                f"(..., {active.total_positions}), got shape {array.shape}"
            )
    if active.length > active.num_active:
        # Padded compact axis: the virtual positions carry zero weights in and out.
        n = active.num_active
        W1_act = np.zeros(W1_np.shape[:2] + (active.length,), dtype=W1_np.dtype)
        W2_act = np.zeros(W2_np.shape[:2] + (active.length,), dtype=W2_np.dtype)
        W1_act[:, :, :n] = W1_np[:, :, active.indices]
        W2_act[:, :, :n] = W2_np[:, :, active.indices]
        return jax.device_put(W1_act), jax.device_put(W2_act)
    W1_act = np.ascontiguousarray(W1_np[:, :, active.indices])
    W2_act = np.ascontiguousarray(W2_np[:, :, active.indices])
    return jax.device_put(W1_act), jax.device_put(W2_act)


def remap_sparse_hh(sparse_hh: Any, active: ActiveSet) -> Any:
    """Rewrite hidden-to-hidden indices from global into compact position space.

    Exactness rests on `active.indices` being sorted: `searchsorted` then returns
    the compact index of any genuinely active position, with no off-by-one. Entries
    that are padding or invalid may map anywhere, so they are clipped into range and
    left to `valid_mask` to zero downstream.

    Args:
        sparse_hh: Connection record with `from_indices`, `to_indices`, `weights`,
            `valid_mask` and `num_valid`.
        active: Active set from `find_active_positions`.

    Returns:
        A record of the same type with indices in compact space.

    Raises:
        ValueError: If the active set is empty, where compact space does not exist.
    """
    if active.num_active == 0:
        raise ValueError(
            "cannot remap into an empty active set; callers must short-circuit "
            "the evaluation instead (every output is sigmoid(0))"
        )
    # Over the padded axis when there is one (a stable shape, one executable per bucket); the
    # virtual positions sort after every real one, so a real position's compact index is the
    # same, and invalid entries clip to the same `num_active - 1` as without padding.
    active_device = jax.device_put(active.search_indices)
    from_compact = jnp.searchsorted(active_device, sparse_hh.from_indices)
    to_compact = jnp.searchsorted(active_device, sparse_hh.to_indices)
    upper = active.num_active - 1
    return type(sparse_hh)(
        from_indices=jnp.clip(from_compact, 0, upper),
        to_indices=jnp.clip(to_compact, 0, upper),
        weights=sparse_hh.weights,
        valid_mask=sparse_hh.valid_mask,
        num_valid=sparse_hh.num_valid,
    )
