"""TensorNEAT's species bookkeeping under JAX's x64 mode.

With ``JAX_ENABLE_X64=1`` (or ``jax.config.update('jax_enable_x64', True)``), two ``lax.cond``
calls in TensorNEAT's ``SpeciesController`` (``tensorneat/algorithm/neat/species.py``) trace
branches whose outputs differ in dtype: ``check_stagnation`` returns the generation and the
species' fitness (float32) on one branch and the species state's ``last_improved`` and
``best_fitness`` (float64) on the other, and ``count_members`` returns ``jnp.nan`` (float64) on
one branch and a float32 count on the other. Tracing stopped at the first ``tell`` with "cond
branches must have equal output types", on every platform, so no run could use x64 in
0.1.0. This module gives the species module a ``jax`` whose ``lax.cond`` casts both branches'
outputs to their promoted dtype, the dtype those calls would carry if both branches agreed;
everything else is the real ``jax``, and TensorNEAT's source is unchanged.

Nothing is patched unless x64 is on when ``ensure_x64_species_dtypes`` runs, so float32 runs
trace exactly what they traced before. The patch is idempotent, and it leaves the species
controller's class alone, so a patch that replaces the class's methods composes with it.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import jax
import jax.numpy as jnp


def unifying_cond(pred: Any, true_fun: Callable, false_fun: Callable, *operands: Any) -> Any:
    """``jax.lax.cond`` whose two branches are cast to the promoted dtype of their outputs.

    Args:
        pred: The condition, as for ``jax.lax.cond``.
        true_fun: The branch taken when ``pred`` is true.
        false_fun: The branch taken otherwise.
        *operands: Passed to the branch, as for ``jax.lax.cond``.

    Returns:
        The taken branch's output, every leaf at the promoted dtype of the two branches' leaves.
    """
    true_out = jax.eval_shape(true_fun, *operands)
    false_out = jax.eval_shape(false_fun, *operands)
    dtypes = jax.tree.map(lambda a, b: jnp.promote_types(a.dtype, b.dtype), true_out, false_out)

    def cast(branch: Callable) -> Callable:
        def run(*ops: Any) -> Any:
            return jax.tree.map(lambda leaf, dtype: jnp.asarray(leaf, dtype=dtype), branch(*ops),
                                dtypes)
        return run

    return jax.lax.cond(pred, cast(true_fun), cast(false_fun), *operands)


class _LaxWithUnifyingCond:
    """``jax.lax`` with ``cond`` replaced by ``unifying_cond``."""

    cond = staticmethod(unifying_cond)

    def __getattr__(self, name: str) -> Any:
        return getattr(jax.lax, name)


class _JaxWithUnifyingCond:
    """The ``jax`` the species module sees under x64: the real one, but for ``lax.cond``."""

    lax = _LaxWithUnifyingCond()

    def __getattr__(self, name: str) -> Any:
        return getattr(jax, name)


def ensure_x64_species_dtypes() -> bool:
    """Give TensorNEAT's species module the dtype-unifying ``lax.cond`` when x64 is on.

    Call it before EMR traces TensorNEAT's ``tell`` (``EMRHyperNEAT.create_config`` does). Under
    float32 it does nothing.

    Returns:
        Whether the species module carries the patch.
    """
    if not jax.config.x64_enabled:
        return False
    from tensorneat.algorithm.neat import species as species_module
    if not isinstance(species_module.jax, _JaxWithUnifyingCond):
        species_module.jax = _JaxWithUnifyingCond()
    return True


def species_module() -> ModuleType:
    """TensorNEAT's species module (for tests)."""
    from tensorneat.algorithm.neat import species
    return species
