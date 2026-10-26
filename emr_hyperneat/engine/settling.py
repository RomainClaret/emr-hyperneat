"""How a recurrent substrate settles: the three settling rules (EMR's, HMR's, additive), the node
activations and node rules, and the step and rule registries that keep their compiled forms."""
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from emr_hyperneat.engine.functions import (
    ACTIVATION_FUNCTIONS,
    hidden_activation,
    scatter_aggregate_by_target,
)


#: Settling steps of HMR's rule for the stable node functions, by (step function, dynamic recurrent
#: input, return_states, number of h->h aggregations, number of step operands), as
#: :data:`_SETTLE_STEPS` for :func:`settle_hidden`.
_SETTLE_STEPS_HMR: Dict[Tuple[Any, ...], Callable] = {}


def _hmr_recurrent_delta(hidden: jnp.ndarray, operands: Tuple[jnp.ndarray, ...], dynamic: bool,
                         num_hh_aggregations: int) -> jnp.ndarray:
    """The hidden-to-hidden input of one step of HMR's rule (:func:`settle_hidden_hmr`)."""
    safe_from, safe_to, effective_weights, *aggregation = operands
    contributions = hidden[:, safe_from] * effective_weights
    if dynamic:
        hh_agg_indices, valid_mask = aggregation
        return scatter_aggregate_by_target(
            contributions, safe_to, hidden.shape[1], hh_agg_indices, valid_mask, num_hh_aggregations
        )
    return jnp.zeros_like(hidden).at[:, safe_to].add(contributions)


def _stable_settle_step_hmr(step_fn: Callable[..., jnp.ndarray], dynamic: bool, return_states: bool,
                            num_hh_aggregations: int, num_step_operands: int = 0) -> Callable:
    """The step of HMR's rule for a stable step function, made once and reused (see
    :func:`_stable_settle_step`): the step function's operands ride in the carry beside the
    recurrent operands."""
    key = (
        step_fn,
        dynamic,
        return_states,
        num_hh_aggregations if dynamic else None,
        num_step_operands,
    )
    step = _SETTLE_STEPS_HMR.get(key)
    if step is None:
        def step(carry, _):
            hidden, *operands = carry
            step_operands, operands = operands[:num_step_operands], operands[num_step_operands:]
            new = step_fn(
                hidden + _hmr_recurrent_delta(hidden, operands, dynamic, num_hh_aggregations),
                *step_operands,
            )
            return (new, *step_operands, *operands), (new if return_states else None)
        _SETTLE_STEPS_HMR[key] = step
    return step


def settle_hidden_hmr(
    drive: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    first_fn: Callable[[jnp.ndarray], jnp.ndarray],
    step_fn: Callable[[jnp.ndarray], jnp.ndarray],
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    return_states: bool = False,
    first_operands: Tuple[jnp.ndarray, ...] = (),
    step_operands: Tuple[jnp.ndarray, ...] = (),
) -> Any:
    """Settle a recurrent substrate's hidden layer by HMR's rule (frozen engine,
    ``forward_hybrid_vmapped`` of the dynamic-functions variants): ``h = first_fn(drive,
    *first_operands)``, then ``activate_time - 1`` steps of ``h = step_fn(h + recurrent(h),
    *step_operands)``.

    The input drive enters at the first step only, and the ``h +`` term acts as a unit
    self-connection on every hidden node, under hidden_only too, so the result differs from
    :func:`settle_hidden` from generation 0 on the same networks (by up to 0.56 of a genome's 1 -
    MSE, part5_settling_probe). The recurrent input is HMR's scatter-add (:func:`settle_hidden` uses
    an equivalent dense matmul, within float32 rounding), so this reproduces the frozen engine's
    arithmetic. A registry step function (tanh) or a node rule (per-node activation,
    :func:`node_activation_rule`) settles through a step made once that carries the rule's operands,
    as in :func:`settle_hidden`, so repeated calls with the same shapes compile nothing new.

    Args:
        drive: (n_samples, num_positions) input drive.
        from_indices, to_indices, hh_weights, valid_mask: the hidden-to-hidden connections.
        activate_time: Settling steps of HMR's count, at least 1 (static): the first activation is
            one of them.
        first_fn: The first activation: the node's configured function.
        step_fn: The activation of the later steps: tanh (HMR's 'initial_only') or the node's
            function ('every_iteration').
        hh_agg_indices, hh_aggregation_mode, num_hh_aggregations, return_states: as for
            :func:`settle_hidden` (the first state is the first activation).
        first_operands, step_operands: The arrays ``first_fn`` and ``step_fn`` read after the
            pre-activation (a node rule's operands); none for a function of the pre-activation
            alone.

    Returns:
        (n_samples, num_positions) settled hidden activity, or ``(hidden, states)`` with
        ``return_states``.
    """
    if activate_time < 1:
        raise ValueError(
            f"activate_time must be at least 1 on the recurrent path, got {activate_time}"
        )
    num_positions = drive.shape[1]
    hidden = first_fn(drive, *first_operands)
    if activate_time == 1:
        return (hidden, hidden[None]) if return_states else hidden
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)
    dynamic = hh_aggregation_mode == 'dynamic' and hh_agg_indices is not None
    operands = (safe_from, safe_to, effective_weights) + (
        (hh_agg_indices, valid_mask) if dynamic else ()
    )

    first = hidden
    step_operands = tuple(step_operands)
    if _is_stable_node_fn(step_fn):
        step = _stable_settle_step_hmr(
            step_fn, dynamic, return_states, num_hh_aggregations, len(step_operands)
        )
        (hidden, *_), states = lax.scan(step, (hidden,) + step_operands + operands, None,
                                        length=activate_time - 1)
    else:
        # Any other step function, typically a partial made per call: caching its step would add an
        # entry per call and never reuse one
        def step(hidden, _):
            new = step_fn(
                hidden + _hmr_recurrent_delta(hidden, operands, dynamic, num_hh_aggregations),
                *step_operands,
            )
            return new, (new if return_states else None)

        hidden, states = lax.scan(step, hidden, None, length=activate_time - 1)
    if return_states:
        return hidden, jnp.concatenate([first[None], states], axis=0)
    return hidden


#: The additive steps (settle_hidden_additive), by update rule, made once like the other steps.
_SETTLE_STEPS_ADDITIVE: Dict[Tuple[Any, ...], Callable] = {}


def _stable_settle_step_additive(update_fn: Callable[..., jnp.ndarray], return_states: bool,
                                 num_update_operands: int) -> Callable:
    """The step of HMR FULL's neuromodulated rule for a stable update rule, made once and reused:
    the rule's operands ride in the carry beside the recurrent operands (see
    :func:`_stable_settle_step_hmr`)."""
    key = (update_fn, return_states, num_update_operands)
    step = _SETTLE_STEPS_ADDITIVE.get(key)
    if step is None:
        def step(carry, _):
            hidden, *operands = carry
            update_operands, operands = (
                operands[:num_update_operands],
                operands[num_update_operands:],
            )
            new = hidden + update_fn(
                _hmr_recurrent_delta(hidden, operands, False, 0), *update_operands
            )
            return (new, *update_operands, *operands), (new if return_states else None)
        step = _SETTLE_STEPS_ADDITIVE.setdefault(key, step)
    return step


def settle_hidden_additive(
    first_hidden: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    update_fn: Callable[..., jnp.ndarray],
    update_operands: Tuple[jnp.ndarray, ...] = (),
    return_states: bool = False,
) -> Any:
    """Settle a neuromodulated recurrent hidden layer by HMR FULL's rule.

    ``h`` starts at the node's value of the input drive (``first_hidden``, computed by the caller),
    then ``activate_time`` steps of ``h = h + update_fn(recurrent(h), *update_operands)``: the
    update is added to the state, where HMR's older rule (:func:`settle_hidden_hmr`) passes ``h +
    recurrent(h)`` through the node and runs ``activate_time - 1`` steps. The recurrent input is
    HMR's scatter-add (:func:`_hmr_recurrent_delta`), summed: FULL's neuromodulated evaluations had
    no per-node hidden-to-hidden aggregation. The update rules are FULL's: under static and context
    gating ``tanh(gates * r)``; under true neuromodulation the node rule with tanh on the
    single-task path and the node itself (the task's activation) on the multi-task path.

    Args:
        first_hidden: (n_samples, num_positions) the node's value of the input drive.
        from_indices, to_indices, hh_weights, valid_mask: the hidden-to-hidden connections.
        activate_time: Settling steps after the first value (static), at least 0.
        update_fn: The update, a node rule of the recurrent input ``update_fn(r,
            *update_operands)``.
        update_operands: The arrays the update reads after the recurrent input.
        return_states: Also return every state, the first value first.

    Returns:
        (n_samples, num_positions) settled hidden activity, or ``(hidden, states)`` with
        ``return_states``.
    """
    if activate_time < 0:
        raise ValueError(f"activate_time must be at least 0, got {activate_time}")
    if activate_time == 0:
        return (first_hidden, first_hidden[None]) if return_states else first_hidden
    num_positions = first_hidden.shape[1]
    operands = (
        jnp.clip(from_indices, 0, num_positions - 1),
        jnp.clip(to_indices, 0, num_positions - 1),
        jnp.where(valid_mask, hh_weights, 0.0),
    )
    update_operands = tuple(update_operands)
    if _is_stable_node_fn(update_fn):
        step = _stable_settle_step_additive(update_fn, return_states, len(update_operands))
        (hidden, *_), states = lax.scan(step, (first_hidden,) + update_operands + operands, None,
                                        length=activate_time)
    else:
        def step(hidden, _):
            new = hidden + update_fn(
                _hmr_recurrent_delta(hidden, operands, False, 0), *update_operands
            )
            return new, (new if return_states else None)

        hidden, states = lax.scan(step, first_hidden, None, length=activate_time)
    if return_states:
        return hidden, jnp.concatenate([first_hidden[None], states], axis=0)
    return hidden


# Recurrent substrate: one settling rule for every recurrent evaluation

#: How the settling steps of the recurrent path activate hidden nodes. ``every_iteration``
#: (the default) applies the configured activation, per node or the one global function, at
#: every step; ``tanh`` settles through tanh whatever the configuration says. There is no
#: ``initial_only``: the forward pass takes no initial activated step, the step the name
#: describes, so under it no configured activation would apply; it is rejected with an
#: explanation.
HH_ACTIVATION_MODES = ('every_iteration', 'tanh')


def node_activation(
    hh_activation_mode: str,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    palette: Optional[jnp.ndarray] = None,
    num_activations: int = 4,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """The activation a settling step applies to hidden nodes.

    Args:
        hh_activation_mode: One of :data:`HH_ACTIVATION_MODES`.
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: The one activation of every node when there are no indices; tanh when
            None.
        palette: Optional ACTIVATION_LIST indices the indices refer to.
        num_activations: Size of the leading ACTIVATION_LIST slice without a palette.

    Returns:
        A function from pre-activation to activation. Without indices it is ``hidden_act_fn``
        itself (tanh when None), which :func:`hidden_activation` would apply alone: the same
        object on every call, so a registry function's settling step is made once
        (:func:`settle_hidden`).

    Raises:
        ValueError: An unknown mode.
    """
    if hh_activation_mode == 'tanh':
        return jnp.tanh
    if hh_activation_mode != 'every_iteration':
        raise ValueError(
            f"hh_activation_mode must be one of {HH_ACTIVATION_MODES}, "
            f"got {hh_activation_mode!r}"
        )
    hidden_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh
    if act_indices is None:
        return hidden_fn
    return functools.partial(
        hidden_activation, act_indices=act_indices, hidden_act_fn=hidden_fn,
        palette=palette, num_activations=num_activations,
    )


#: Node functions whose settling step is made once and reused: the activation registry's
#: functions, which stay the same objects for the life of the module (``jnp.tanh`` among them).
_STABLE_NODE_FNS = tuple(ACTIVATION_FUNCTIONS.values())

#: Node rules, made once per setting and reused like the registry's functions. A rule takes the
#: pre-activation and then the per-genome arrays it reads, ``rule(pre, *operands)``, which the
#: settling step carries (``node_operands`` of :func:`settle_hidden`). Keyed by the settings a rule
#: closes over, so bounded by the configurations a process runs.
_NODE_RULES: Dict[Tuple[Any, ...], Callable] = {}

#: Settling steps of the stable node functions, by (node function, dynamic recurrent input,
#: return_states, number of h->h aggregations, number of node operands); bounded by the registry,
#: the node rules and those settings.
_SETTLE_STEPS: Dict[Tuple[Any, ...], Callable] = {}


def _is_stable_node_fn(node_fn: Callable) -> bool:
    """A registry function or a node rule: the same object on every call."""
    return (any(node_fn is fn for fn in _STABLE_NODE_FNS)
            or any(node_fn is rule for rule in _NODE_RULES.values()))


def _node_rule(key: Tuple[Any, ...], make: Callable[[], Callable],
               inner: Optional[Callable] = None) -> Callable:
    """The node rule of ``key``, made by ``make`` the first time.

    A rule built on another node function (``inner``) is kept only when that function is stable:
    over a function made per call it would be a new rule on every call, and keeping it would add
    an entry per call and never reuse one.
    """
    if inner is not None and not _is_stable_node_fn(inner):
        return make()
    rule = _NODE_RULES.get(key)
    if rule is None:
        # setdefault: a rule another thread kept meanwhile is the one every caller shares
        rule = _NODE_RULES.setdefault(key, make())
    return rule


def node_activation_rule(
    hh_activation_mode: str,
    act_indices: Optional[jnp.ndarray] = None,
    hidden_act_fn: Optional[Callable] = None,
    palette: Optional[jnp.ndarray] = None,
    num_activations: int = 4,
) -> Tuple[Callable, Tuple[jnp.ndarray, ...]]:
    """:func:`node_activation` as a node rule and the arrays it reads.

    Args:
        hh_activation_mode: One of :data:`HH_ACTIVATION_MODES`.
        act_indices: Optional (num_positions,) per-node activation indices.
        hidden_act_fn: The one activation of every node when there are no indices; tanh when
            None.
        palette: Optional ACTIVATION_LIST indices the indices refer to (concrete).
        num_activations: Size of the leading ACTIVATION_LIST slice without a palette.

    Returns:
        ``(rule, operands)``, ``rule(pre, *operands)`` being ``node_activation(...)(pre)``.
        Without indices, or under ``'tanh'``, the rule is that function itself and there are no
        operands. With indices it is the per-node activation of :func:`hidden_activation`, made
        once per palette and number of activations, and the indices are its operand: a settling
        step on it is made once and carries them (:func:`settle_hidden`), where the partial of
        :func:`node_activation` is a new object on every call, and so is its step.

    Raises:
        ValueError: An unknown mode.
    """
    whole = node_activation(hh_activation_mode, None, hidden_act_fn)
    if act_indices is None or hh_activation_mode == 'tanh':
        return whole, ()
    entries = None if palette is None else tuple(int(i) for i in np.asarray(palette).tolist())

    def make():
        def per_node_activation(pre, indices):
            return hidden_activation(pre, indices, palette=entries,
                                     num_activations=num_activations)
        return per_node_activation

    return _node_rule(('activation', entries, num_activations), make), (act_indices,)


def _recurrent_input(hidden: jnp.ndarray, operands: Tuple[jnp.ndarray, ...], dynamic: bool,
                     num_hh_aggregations: int) -> jnp.ndarray:
    """The hidden-to-hidden input of one settling step (see :func:`settle_hidden`)."""
    if dynamic:
        safe_from, safe_to, effective_weights, hh_agg_indices, valid_mask = operands
        return scatter_aggregate_by_target(
            hidden[:, safe_from] * effective_weights, safe_to, hidden.shape[1],
            hh_agg_indices, valid_mask, num_hh_aggregations
        )
    (Whh,) = operands
    return hidden @ Whh


def _stable_settle_step(node_fn: Callable[..., jnp.ndarray], dynamic: bool,
                        return_states: bool, num_hh_aggregations: int,
                        num_node_operands: int = 0) -> Callable:
    """The settling step of a stable node function, made once and reused.

    ``lax.scan`` traces its step through a cache keyed on the step's function object, and the
    generation's evaluation runs the scan under ``vmap`` without ``jit``, which compiles it
    once per traced step. A step made inside every call is a new object every time, so the
    scan is traced and compiled again on every call, once per generation. This one takes the
    drive, the node function's operands and the recurrent operands from the carry and closes
    over nothing that changes between calls, so a run compiles it once per shape.
    """
    key = (node_fn, dynamic, return_states, num_hh_aggregations if dynamic else None,
           num_node_operands)
    step = _SETTLE_STEPS.get(key)
    if step is None:
        def step(carry, _):
            hidden, drive, *operands = carry
            node_operands, operands = operands[:num_node_operands], operands[num_node_operands:]
            new = node_fn(drive + _recurrent_input(hidden, operands, dynamic, num_hh_aggregations),
                          *node_operands)
            return (new, drive, *node_operands, *operands), (new if return_states else None)
        _SETTLE_STEPS[key] = step
    return step


def settle_hidden(
    drive: jnp.ndarray,
    from_indices: jnp.ndarray,
    to_indices: jnp.ndarray,
    hh_weights: jnp.ndarray,
    valid_mask: jnp.ndarray,
    activate_time: int,
    node_fn: Callable[..., jnp.ndarray],
    hh_agg_indices: Optional[jnp.ndarray] = None,
    hh_aggregation_mode: str = 'sum',
    num_hh_aggregations: int = 4,
    return_states: bool = False,
    node_operands: Tuple[jnp.ndarray, ...] = (),
) -> Any:
    """Settle a recurrent substrate's hidden layer.

    The one rule of every recurrent evaluation: ``h(0) = 0`` and, for ``activate_time``
    steps, ``h(t+1) = node_fn(drive + recurrent(h(t)), *node_operands)``. ``drive`` is the
    input drive of :func:`input_drive`, constant across steps and re-injected at each;
    ``recurrent`` sums the hidden-to-hidden contributions (or aggregates them per node in
    ``hh_aggregation_mode='dynamic'``); ``node_fn`` is the node's whole transformation,
    activation and any neuromodulation, the same one the feed-forward evaluation applies. With
    no hidden-to-hidden connection every step returns ``node_fn(drive, *node_operands)``, so
    the recurrent path computes the feed-forward substrate's arithmetic. On CPU the two agree
    bit for bit, except for compound activations such as ``resonator`` and ``osc_adapt``, which
    XLA can lower differently inside the settling loop: those differ by a float32 ulp.

    A node function of the activation registry (``jnp.tanh`` on the plain recurrent path, the
    configured function under a global activation, :func:`node_activation`) or a node rule (per-node
    activation, :func:`node_activation_rule`, and the neuromodulated nodes built on it,
    :func:`gated_node_rule`, :func:`modulatory_node_rule` (every modulatory-neuron option
    combination) and :func:`true_neuromodulation_node_rule`) settles through
    a step made once (:func:`_stable_settle_step`), which carries the rule's operands, so repeated
    calls with the same shapes compile nothing new. Any other node function, such as a partial or a
    closure made per call, gets a step made per call and is traced and compiled again on every call.

    Args:
        drive: (n_samples, num_positions) input drive.
        from_indices: (max_sparse_conns,) source indices.
        to_indices: (max_sparse_conns,) target indices.
        hh_weights: (max_sparse_conns,) weights.
        valid_mask: (max_sparse_conns,) validity mask.
        activate_time: Number of settling steps, at least 1 (static).
        node_fn: Pre-activation (then ``node_operands``) to activation, per node.
        hh_agg_indices: Per-node hidden-to-hidden aggregation indices, for
            ``hh_aggregation_mode='dynamic'``.
        hh_aggregation_mode: ``'sum'`` or ``'dynamic'``.
        num_hh_aggregations: Number of aggregation functions for the dynamic mode.
        return_states: Also return the hidden state after each step (the substrate API's settling view);
            the settled state is computed the same way either way.
        node_operands: The arrays ``node_fn`` reads after the pre-activation (a node rule's
            operands); none for a function of the pre-activation alone.

    Returns:
        (n_samples, num_positions) settled hidden activity; with ``return_states`` the pair
        ``(hidden, states)``, states (activate_time, n_samples, num_positions), the last equal to hidden.

    Raises:
        ValueError: ``activate_time`` below 1, which would return the zero state.
    """
    if activate_time < 1:
        raise ValueError(
            f"activate_time must be at least 1 on the recurrent path, got {activate_time}"
        )
    num_positions = drive.shape[1]
    hidden = jnp.zeros((drive.shape[0], num_positions))
    safe_from = jnp.clip(from_indices, 0, num_positions - 1)
    safe_to = jnp.clip(to_indices, 0, num_positions - 1)
    effective_weights = jnp.where(valid_mask, hh_weights, 0.0)

    dynamic = hh_aggregation_mode == 'dynamic' and hh_agg_indices is not None
    if dynamic:
        operands = (safe_from, safe_to, effective_weights, hh_agg_indices, valid_mask)
    else:
        # The sparse h->h weights as one dense matrix (Whh[i, j] = sum of weights i->j), so
        # each settling step is a matmul instead of a per-step scatter-add, the CPU bottleneck
        # that made recurrent topologies look infeasible. Equal to the scatter-add up to
        # float32 rounding (max abs diff ~5e-7); all zero without h->h connections.
        operands = (jnp.zeros((num_positions, num_positions)).at[safe_from, safe_to].add(
            effective_weights
        ),)

    node_operands = tuple(node_operands)
    if _is_stable_node_fn(node_fn):
        step = _stable_settle_step(node_fn, dynamic, return_states, num_hh_aggregations,
                                   len(node_operands))
        (hidden, *_), states = lax.scan(step, (hidden, drive) + node_operands + operands, None,
                                        length=activate_time)
    else:
        # Any other node function, typically a partial or closure made per call: caching its
        # step would add an entry per call and never reuse one
        def step(hidden, _):
            new = node_fn(drive + _recurrent_input(hidden, operands, dynamic,
                                                   num_hh_aggregations), *node_operands)
            return new, (new if return_states else None)

        hidden, states = lax.scan(step, hidden, None, length=activate_time)
    if return_states:
        return hidden, states
    return hidden
