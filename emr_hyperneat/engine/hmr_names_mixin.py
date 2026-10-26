"""The engine's HMR names: the attribute and property names HMR runners read, its result keys, and the
multi-task methods it named that EMR refuses with a pointer to its own."""
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp

from emr_hyperneat.engine.config import EMRConfig
from emr_hyperneat.engine.metrics import EMRRecurrenceMetrics
from emr_hyperneat.execution import ExecutionMode


class HMRNamesMixin:
    """EMRHyperNEAT's HMR-compatible names."""

    @property
    def _hidden_act_func(self) -> Optional[Callable]:
        """HMR's name for _hidden_act_fn, the one hidden activation of a non-per-node run (None when
        per node)."""
        return self._hidden_act_fn

    @_hidden_act_func.setter
    def _hidden_act_func(self, value: Optional[Callable]) -> None:
        self._hidden_act_fn = value

    @property
    def _output_act_func(self) -> Optional[Callable]:
        """HMR's name for _output_act_fn, the output activation."""
        return self._output_act_fn

    @_output_act_func.setter
    def _output_act_func(self, value: Optional[Callable]) -> None:
        self._output_act_fn = value

    @property
    def aggregation_mode(self) -> str:
        """HMR's name for agg_mode (its aggregation variant's attribute)."""
        return self.agg_mode

    @property
    def _depth_offload_enabled(self) -> bool:
        """HMR's name (its depth-offload variant): whether this configuration runs the depth-offload
        path."""
        return self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD

    @property
    def _use_sparse_hh(self) -> bool:
        """HMR's unified model's flag: hidden-to-hidden connections are discovered."""
        return self.extended_config is not None and bool(
            self.extended_config.allow_hidden_to_hidden
        )

    @property
    def _use_full_recurrent(self) -> bool:
        """HMR's unified model's flag, by its rule (get_recurrence_mode() == 'full_recurrent'):
        recurrence enabled with backward, lateral or self connections."""
        return self.get_recurrence_mode() == 'full_recurrent'

    def get_recurrence_metrics(self) -> Optional[EMRRecurrenceMetrics]:
        """HMR's unified model's name for the last generation's recurrence metrics
        (get_extended_metrics; the record carries every field of HMR's RecurrenceMetrics)."""
        return self._extended_metrics

    def _hmr_result_keys(
        self, target_fitness: float, last_best: float, best: float
    ) -> Dict[str, Any]:
        """HMR's result keys of run_until_threshold (its streaming and depth-offload variants):
        'converged' and its alias 'solved' (the run reached target_fitness), 'final_fitness' (the
        last generation's best, where 'best_fitness' is the run's best) and 'sparse_hh_stats' (the
        last generation's hidden-to-hidden discovery: its time, its connections over the population
        and their mean per genome)."""
        reached = bool(best >= target_fitness)
        metrics = self._extended_metrics
        stats = None
        if metrics is not None:
            connections = int(metrics.hidden_to_hidden_connections)
            stats = {
                'discovery_time_ms': float(metrics.time_hh_discovery_ms),
                'hh_connections': connections,
                'total_connections': connections,
                'avg_connections_per_genome': connections
                / max(int(getattr(self, 'population_size', 1)), 1),
            }
        return {
            'converged': reached,
            'solved': reached,
            'final_fitness': float(last_best),
            'sparse_hh_stats': stats,
        }

    # Execution mode: streaming and depth offload

    @property
    def _recurrence_config(self) -> Optional['EMRConfig']:
        """Alias for `extended_config`, the single source of recurrence truth.

        A separate `_recurrence_config` for depth offload, set only by an imperative
        `set_recurrence_config` call that `create_config` never makes, would leave a
        YAML-configured recurrent run reading an unset config and evaluating feedforward,
        with no hidden-to-hidden connection. Aliasing leaves exactly one place for the
        answer to live.
        """
        return self.extended_config

    @property
    def cppn_num_outputs(self) -> int:
        """HMR's name for ``num_cppn_outputs``, the CPPN's output count."""
        return self.num_cppn_outputs

    def evaluate_with_task_context(
        self, state: Any, task_context: Any, inputs: Any, targets: Any
    ) -> jnp.ndarray:
        """HMR's name (its neuromodulation variant), refused: there it read _cached_G1_logit, which
        nothing assigned, so it raised before evaluating anything, and no runner calls it.

        Raises:
            NotImplementedError: Always; context gating takes its context from the current task
                index (set_task_neurotransmitter), and run_generation_multitask evaluates several
                tasks.
        """
        raise NotImplementedError(
            "evaluate_with_task_context is HMR's, where it never ran (it read gate logits nothing "
            "set); context gating takes its context from the task index "
            "(set_task_neurotransmitter), and run_generation_multitask evaluates several tasks")

    def evaluate_multi_task(
        self, state: Any, multi_task_problem: Any, aggregation: str = 'mean'
    ) -> jnp.ndarray:
        """HMR's name, refused for the reason evaluate_with_task_context gives (it called that
        method).

        Raises:
            NotImplementedError: Always; run_generation_multitask evaluates several tasks.
        """
        raise NotImplementedError(
            "evaluate_multi_task is HMR's, where it never ran (it called "
            "evaluate_with_task_context); use "
            "run_generation_multitask")

    def run_multi_task_generation(
        self, state: Any, multi_task_problem: Any, aggregation: str = 'mean'
    ) -> Any:
        """HMR's name, refused for the reason evaluate_with_task_context gives (it called that
        method, and it would have told the new population the previous population's fitnesses).

        Raises:
            NotImplementedError: Always; run_generation_multitask runs a multi-task generation.
        """
        raise NotImplementedError(
            "run_multi_task_generation is HMR's, where it never ran (it called "
            "evaluate_with_task_context); use "
            "run_generation_multitask")
