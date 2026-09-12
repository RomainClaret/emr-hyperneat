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
arbitrary positions, so no chunk is self-contained. Compaction is the way out.

Rather than revert the dense `Whh` -- it replaced a scatter-add that was the CPU
bottleneck making recurrent topologies look infeasible -- shrink the *space* it is
built over. Evaluate on the union of positions any genome actually reached, and the
matmul stays dense and fast while the matrix becomes negligible.

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

__all__ = ["ActiveSet", "find_active_positions", "compact_weights", "remap_sparse_hh"]


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
    and it is worth stating because the *evaluator* consumes W2 transposed, as
    `(pop, positions, n_out)`. Gathering on the wrong axis here would silently index
    the output dimension instead of the position dimension.

    Args:
        W1: Input-to-hidden weights, (pop, n_in, total_positions).
        W2: Hidden-to-output weights, (pop, n_out, total_positions) -- host layout,
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
    active_device = jax.device_put(active.indices)
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
