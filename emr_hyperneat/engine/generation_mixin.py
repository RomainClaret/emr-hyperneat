"""One generation: run_generation, the standard generation (run_generation_verbose) and the compiled
feed-forward core, the query and evaluation backends, the forward built per configuration, and the
records a generation leaves."""
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat._compat.core.base_algorithm import AlgorithmMetrics
from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid
from emr_hyperneat.engine.config import ForwardPassMode, SparseHiddenConnections
from emr_hyperneat.engine.discovery import (
    _last_discovery_stats,
    discover_sparse_hh_vectorized_multi_hop,
)
from emr_hyperneat.engine.evaluation import (
    EvaluatedGeneration,
    FeedforwardGeneration,
    _auto_eval_chunk_size,
    _plain_recurrent_forward,
    evaluate_population_on_device,
)
from emr_hyperneat.engine.functions import (
    AGGREGATION_FUNCTIONS,
    AGGREGATION_LIST,
    activation_interpretation_signal,
    aggregation_from_weight_interpretation,
    continuous_to_index,
    cppn_output_function_indices,
    dense_substrate_forward,
    input_drive,
    level_sparsity_quantiles,
    node_sparsity_quantiles,
    random_function_indices,
    sparsify_by_quantile,
    sparsify_winner_take_all,
    stp_depression,
)
from emr_hyperneat.engine.leo import LEO_CHANNEL, leo_sparsify
from emr_hyperneat.engine.metrics import (
    EMRRecurrenceMetrics,
    fitness_statistics,
    sanitize_fitness,
    sanitize_fitness_on_device,
    species_summary,
    topology_summary,
)
from emr_hyperneat.engine.neuromodulation import (
    _neuromodulated_rule,
    apply_gate_scaling,
    cppn_modulatory_masks,
    derive_context_from_input,
    derive_receptor_from_weight,
    gated_node_rule,
    invert_outputs,
    modulatory_masks,
    modulatory_neuron_masks,
    modulatory_node_rule,
    true_neuromodulation_node_rule,
)
from emr_hyperneat.engine.parallel import evaluate_population_split, query_population_split
from emr_hyperneat.engine.settling import (
    node_activation_rule,
    settle_hidden,
    settle_hidden_additive,
    settle_hidden_hmr,
)
from emr_hyperneat.engine.substrate_build import (
    apply_locality_penalty_to_weights,
    batch_query_population_multi_source_multi_output,
    batch_query_population_self_connections,
    query_population_on_device,
    substrate_masks,
    substrate_query_rows,
    substrate_weight_matrices,
)
from emr_hyperneat.execution import ExecutionMode


class GenerationMixin:
    """EMRHyperNEAT's generation."""

    def run_generation(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        verbose: bool = True,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation, dispatching on execution mode and multi-task.

        Two independent dispatches happen here. Execution mode comes first because
        depth offload fuses matrix building with evaluation and so replaces the
        whole generation rather than a step inside it; streaming and standard both
        continue into the normal loop, which selects the build step itself.

        Args:
            state: Algorithm state
            problem: Problem instance (or MultiTaskProblem for multi-task)
            skip_metrics: Depth offload then reports 0 for the mean, min and std fitness; the
                other execution modes compute every metric whatever it says
            verbose: Verbose output

        Returns:
            Tuple of (new_state, metrics)
        """
        if self.network_model == 'node_list':
            # The node-list model has one execution path and one task, so neither dispatch below
            # applies (its configuration refuses both)
            return self.run_generation_verbose(state, problem, skip_metrics)

        if self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD:
            self._refuse_palette_evolution('depth offload')
            # Imported here rather than at module scope: the strategy modules import from
            # emrhyperneat, which imports this module, so a top-level import would form a cycle.
            from emr_hyperneat.execution.offload import (
                run_generation_offload,
            )
            return run_generation_offload(self, state, problem, skip_metrics)

        # Check if multi-task mode is enabled
        if (self.multitask_config is not None and
            getattr(self.multitask_config, 'enabled', False)):
            # Multi-task mode: extract individual problems from MultiTaskProblem
            if hasattr(problem, 'tasks'):
                problems = problem.tasks
            elif hasattr(problem, 'problems'):
                problems = problem.problems
            else:
                problems = [problem]

            # Get NT vectors from state
            neurotransmitters = self._multitask_state.get('nt_vectors', None)

            return self.run_generation_multitask(
                state,
                problems,
                neurotransmitters,
                aggregation_method=self.multitask_config.fitness_aggregation,
                aggregation_weights=self.multitask_config.task_weights,
                orthogonality_bonus=self.multitask_config.orthogonality_bonus,
                specialization_bonus=self.multitask_config.specialization_bonus,
            )
        else:
            # Single-task mode: use standard implementation
            return self.run_generation_verbose(state, problem, skip_metrics)

    def _build_forward(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        sparse_hh: Optional['SparseHiddenConnections'],
        neuromod_data: Optional[Dict[str, Any]],
        act_indices: Optional[jnp.ndarray],
        hh_agg_indices: Optional[jnp.ndarray],
        ih_agg_indices: Optional[jnp.ndarray],
        ih_global_agg_fn: Optional[Callable],
        use_dynamic_eval: bool,
        use_ih_aggregation: bool,
        pop_size: int,
        total_positions: int,
        node_masks: Optional[jnp.ndarray] = None,
        sparsity_quantiles: Optional[jnp.ndarray] = None,
        wta_k_percent: Optional[float] = None,
    ) -> Tuple[Callable, Tuple, bool]:
        """This generation's per-genome forward and the population arrays it runs on.

        One function per configuration serves the population's fitness and
        :meth:`evaluate_on_data`, so a held-out score describes the network that was evolved.
        Feed-forward and recurrent evaluation share every per-node transformation (input
        drive, activation, neuromodulation); the recurrent one settles it with
        :func:`settle_hidden`, so without hidden-to-hidden connections the two compute the
        same arithmetic (bit for bit on CPU but for the exception :func:`settle_hidden` names).

        Args:
            W1, W2: Population weight matrices as evaluated (compacted on the recurrent path).
            sparse_hh: Hidden-to-hidden connections, or None for feed-forward evaluation.
            neuromod_data: The generation's neuromodulation arrays, or None.
            act_indices: Per-node activation indices, or None.
            hh_agg_indices: Per-node hidden-to-hidden aggregation indices, or None.
            ih_agg_indices: Per-node input-to-hidden aggregation indices, or None.
            ih_global_agg_fn: One input-to-hidden aggregation for every node, or None.
            use_dynamic_eval: A dynamic-functions mode is on.
            use_ih_aggregation: An input-to-hidden aggregation mode is on.
            pop_size: Genomes evaluated.
            total_positions: Positions evaluated.
            node_masks: The active positions (pop, positions); read only with sparsity.
            sparsity_quantiles: Per-node quantile below which a hidden value is zeroed (pop,
                positions), or None.
            wta_k_percent: The fraction of active values winner-take-all keeps, or None. Sparsity is
                applied by the feed-forward dynamic evaluation only (configuration refuses it
                elsewhere).

        Returns:
            ``(forward_one, genome_args, chunk_by_scan)``. ``forward_one(inputs, *args)`` is
            one genome's outputs; ``genome_args`` are the population arrays (population axis
            first, None where unused); ``chunk_by_scan`` selects the scan-based chunked
            evaluation, which the plain recurrent path has always used.
        """
        recurrent = sparse_hh is not None
        palette = self.df_palette
        num_activations = self.df_num_activations
        hidden_fn = self._hidden_act_fn if self._hidden_act_fn is not None else jnp.tanh
        output_fn = self._output_act_fn if self._output_act_fn is not None else jax.nn.sigmoid
        activate_time = self.activate_time
        hh_mode = self.hh_activation_mode
        hh_agg_mode = self.hh_aggregation_mode
        # Evaluation dispatches over AGGREGATION_LIST[:count]; a palette maps into the whole list
        num_hh_agg = (
            len(AGGREGATION_LIST) if self.agg_palette is not None else self.agg_num_aggregations
        )
        per_node_agg = ih_agg_indices is not None
        agg_args = (ih_agg_indices if per_node_agg
                    else jnp.zeros((pop_size, total_positions), dtype=jnp.int32))
        drive_kwargs = dict(
            global_agg_fn=ih_global_agg_fn,
            per_node_aggregation=per_node_agg,
            true_aggregation=self.agg_use_true_aggregation,
            num_aggregations=num_hh_agg,
        )
        hh_args: Tuple = ()
        if recurrent:
            hh_args = (sparse_hh.from_indices, sparse_hh.to_indices, sparse_hh.weights,
                       sparse_hh.valid_mask, hh_agg_indices)

        def activation(act_idx):
            # A settling step applies hh_activation_mode's activation; a dense pass the
            # configured one.
            if recurrent and hh_mode == 'initial_only':
                return jnp.tanh, ()  # HMR's later steps (settling_rule 'hmr' only)
            return node_activation_rule(hh_mode if recurrent else 'every_iteration', act_idx,
                                        hidden_fn, palette, num_activations)

        def first_activation(act_idx):
            # HMR's rule applies the configured activation at its first step ('tanh' mode: tanh
            # there too)
            if hh_mode == 'tanh':
                return jnp.tanh, ()
            return node_activation_rule(
                'every_iteration', act_idx, hidden_fn, palette, num_activations
            )

        settling_rule = self._settling_rule

        def hidden_state(inputs, W1_i, agg_idx, node, hh, first=None, update=None):
            node_fn, node_operands = node
            drive = input_drive(inputs, W1_i, agg_idx, **drive_kwargs)
            if not recurrent:
                return node_fn(drive, *node_operands)
            from_idx, to_idx, hh_w, valid, hh_agg_idx = hh
            if settling_rule == 'hmr' and update is not None:
                # HMR FULL's neuromodulated rule: the node of the drive with the configured
                # activation (first), then activate_time steps of h + update(recurrent(h))
                first_fn, first_operands = first
                update_fn, update_operands = update
                return settle_hidden_additive(
                    first_fn(drive, *first_operands),
                    from_idx,
                    to_idx,
                    hh_w,
                    valid,
                    activate_time,
                    update_fn,
                    update_operands,
                )
            if settling_rule == 'hmr':
                # HMR's rule: the first step applies the configured activation (first), the later
                # ones the settling activation (node: tanh under 'initial_only' or 'tanh')
                first_fn, first_operands = first if first is not None else node
                return settle_hidden_hmr(drive, from_idx, to_idx, hh_w, valid, activate_time,
                                         first_fn, node_fn, hh_agg_idx, hh_agg_mode, num_hh_agg,
                                         first_operands=first_operands, step_operands=node_operands)
            return settle_hidden(drive, from_idx, to_idx, hh_w, valid, activate_time, node_fn,
                                 hh_agg_idx, hh_agg_mode, num_hh_agg, node_operands=node_operands)

        if neuromod_data is not None:
            mode = neuromod_data['mode']
            if mode == 'static_gating':
                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, gates, *hh):
                    node = gated_node_rule(activation(act_idx), gates)
                    first, update = (
                        (
                            gated_node_rule(first_activation(act_idx), gates),
                            (_neuromodulated_rule('gain', jnp.tanh), (gates,)),
                        )
                        if recurrent and settling_rule == 'hmr'
                        else (None, None)
                    )
                    h = hidden_state(inputs, W1_i, agg_idx, node, hh, first, update)
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['static_gates']) + hh_args, False

            if mode == 'context_gating' and neuromod_data.get('gate_source') == 'cppn':
                # HMR's Level 2: each case gates every connection with sigmoid(logit + influence *
                # mean of the case's input statistics), the context HMR derived (mean, std, min, max
                # of the input vector)
                context_influence = neuromod_data['context_influence']

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, logit_1, logit_2):
                    shift = context_influence * jnp.mean(
                        jax.vmap(lambda case: derive_context_from_input(case, 'statistics'))(
                            inputs
                        ),
                        axis=1,
                    )
                    W1_case = W1_i[None] * jax.nn.sigmoid(logit_1[None] + shift[:, None, None])
                    W2_case = W2_i[None] * jax.nn.sigmoid(logit_2[None] + shift[:, None, None])
                    act_fn, act_operands = activation(act_idx)
                    h = act_fn(jnp.einsum('ci,cip->cp', inputs, W1_case), *act_operands)
                    return output_fn(jnp.einsum('cp,cpo->co', h, W2_case))
                return forward_one, (W1, W2, act_indices, agg_args, neuromod_data['G1_logit'],
                                     neuromod_data['G2_logit']), False

            if mode == 'context_gating':
                context_influence = neuromod_data['context_influence']
                context_idx = self._multitask_state.get('current_task_idx', 0)

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, base_g, ctx_mod, *hh):
                    gates = jax.nn.sigmoid(base_g + context_influence * ctx_mod[context_idx])
                    node = gated_node_rule(activation(act_idx), gates)
                    first, update = (
                        (
                            gated_node_rule(first_activation(act_idx), gates),
                            (_neuromodulated_rule('gain', jnp.tanh), (gates,)),
                        )
                        if recurrent and settling_rule == 'hmr'
                        else (None, None)
                    )
                    h = hidden_state(inputs, W1_i, agg_idx, node, hh, first, update)
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['base_gates'],
                                     neuromod_data['context_modulation']) + hh_args, False

            if mode == 'modulatory_neurons':
                # One node rule for every combination of the modulatory options: the static values
                # in its key, the genome's arrays carried (the per-node weights, the factors, then
                # the mask; the two sources' genome arguments are unchanged). The signal is each
                # case's own (HMR used the first case's activity for every case) unless a
                # matched-gain control replaces it.
                mod_type = neuromod_data['mod_connection_type']
                signal, node_weights, scope, matched_gain = neuromod_data['modulatory_options']
                mod_decay = neuromod_data['mod_decay']
                carried = (
                    (
                        (neuromod_data['mod_connection_weights'],)
                        if node_weights == 'variance'
                        else ()
                    )
                    + ((neuromod_data['mod_factor'],) if signal == 'sum_factor' else ())
                    + (neuromod_data['mod_mask'],)
                )
                num_carried = len(carried)

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, *rest):
                    node = modulatory_node_rule(
                        activation(act_idx),
                        rest[:num_carried],
                        mod_type,
                        signal,
                        node_weights,
                        scope,
                        mod_decay,
                        matched_gain,
                    )
                    h = hidden_state(inputs, W1_i, agg_idx, node, rest[num_carried:])
                    return output_fn(h @ W2_i)
                return forward_one, (W1, W2, act_indices, agg_args) + carried + hh_args, False

            if mode == 'true_neuromodulation':
                neurotransmitter = neuromod_data['neurotransmitter']
                mod_strength = neuromod_data['mod_strength']
                modulation_mode = neuromod_data['modulation_mode']
                use_output_inversion = neuromod_data['use_output_inversion']
                num_nt = neuromod_data['num_nt']
                node_rule = neuromod_data['node_rule']

                def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, receptors, base_g, *hh):
                    node = true_neuromodulation_node_rule(activation(act_idx), receptors, base_g,
                                                          neurotransmitter, mod_strength, num_nt,
                                                          modulation_mode, node_rule)
                    first, update = (
                        (
                            true_neuromodulation_node_rule(
                                first_activation(act_idx),
                                receptors,
                                base_g,
                                neurotransmitter,
                                mod_strength,
                                num_nt,
                                modulation_mode,
                                node_rule,
                            ),
                            true_neuromodulation_node_rule(
                                (jnp.tanh, ()),
                                receptors,
                                base_g,
                                neurotransmitter,
                                mod_strength,
                                num_nt,
                                modulation_mode,
                                node_rule,
                            ),
                        )
                        if recurrent and settling_rule == 'hmr'
                        else (None, None)
                    )
                    h = hidden_state(inputs, W1_i, agg_idx, node, hh, first, update)
                    return invert_outputs(output_fn(h @ W2_i), neurotransmitter,
                                          use_output_inversion, num_nt, node_rule)
                return forward_one, (W1, W2, act_indices, agg_args,
                                     neuromod_data['receptor_densities'],
                                     neuromod_data['base_gains']) + hh_args, False

            raise ValueError(f"Unknown neuromodulation mode: {mode}")

        dynamic = ((use_dynamic_eval and (act_indices is not None
                                          or self._hidden_act_fn is not None))
                   or use_ih_aggregation or (recurrent and hh_agg_indices is not None))
        if (
            dynamic
            and not recurrent
            and (sparsity_quantiles is not None or wta_k_percent is not None)
        ):
            def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, mask_i, quantiles_i):
                h = hidden_state(inputs, W1_i, agg_idx, activation(act_idx), ())
                if wta_k_percent is not None:
                    h = sparsify_winner_take_all(h, wta_k_percent, mask_i)
                else:
                    h = sparsify_by_quantile(h, quantiles_i, mask_i)
                return output_fn(h @ W2_i)
            return (
                forward_one,
                (W1, W2, act_indices, agg_args, node_masks, sparsity_quantiles),
                False,
            )
        if dynamic and not recurrent:
            def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx):
                return dense_substrate_forward(
                    inputs, W1_i, W2_i, act_idx, agg_idx,
                    hidden_act_fn=hidden_fn, output_act_fn=output_fn, palette=palette,
                    num_activations=num_activations, **drive_kwargs,
                )
            return forward_one, (W1, W2, act_indices, agg_args), False
        if dynamic:
            def forward_one(inputs, W1_i, W2_i, act_idx, agg_idx, *hh):
                h = hidden_state(inputs, W1_i, agg_idx, activation(act_idx), hh,
                                 first=first_activation(act_idx))
                return output_fn(h @ W2_i)
            return forward_one, (W1, W2, act_indices, agg_args) + hh_args, False
        if not recurrent:
            def forward_one(inputs, W1_i, W2_i):
                # Fixed tanh hidden and sigmoid output, whatever output_activation says
                return dense_substrate_forward(inputs, W1_i, W2_i,
                                               hidden_act_fn=jnp.tanh,
                                               output_act_fn=jax.nn.sigmoid)
            return forward_one, (W1, W2), False

        # The plain recurrent forward (HMR's under settling_rule 'hmr'), the same object every
        # generation, so the scan-chunked evaluation compiles once per shape
        return _plain_recurrent_forward(activate_time, settling_rule), (W1, W2) + hh_args[:4], True

    def _record_evaluation(
        self,
        forward_one: Callable,
        genome_args: Tuple,
        fitnesses: jnp.ndarray,
        cppn_population: Tuple,
        masks_full: jnp.ndarray,
        masks_eval: jnp.ndarray,
        eval_positions: Optional[jnp.ndarray],
        act_indices: Optional[jnp.ndarray],
        ih_agg_indices: Optional[jnp.ndarray],
        next_state: Any,
        full_weights: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Keep what :meth:`evaluate_on_data` needs to re-run this generation's champion.

        The champion is the genome ``next_state`` carries first: TensorNEAT places the best
        genome of a generation, unmutated, first in the next population. Under a tie for the
        best fitness its sorts keep the highest index of the top species, not the first
        index ``argmax`` returns, so a tie is resolved by comparing genomes.

        Only the champion's slices are kept, twice (a leading axis of two): XLA lowers a
        batch of one genome to other reduction kernels, which moved the fitness by one
        float32 ulp, while two copies lower like the population batch. On CPU the
        training-data score then reproduces the champion's fitness bit for bit.

        ``full_weights`` names the population arrays whose champion slices the record keeps for
        the substrate statistics: by default ``W1_full`` and ``W2_full`` from the cached W1 and
        W2; the node-list model passes its blocks, ``W_hh_full`` among them.
        """
        best = np.asarray(fitnesses)
        tied = np.flatnonzero(best == best.max())
        champion = int(tied[0])
        if tied.size > 1:
            kept = self._compiled_ask(next_state)
            same = np.ones(tied.size, dtype=bool)
            for part, kept_part in zip(cppn_population[:2], kept[:2]):
                rows = np.asarray(part)[tied]
                one = np.asarray(kept_part[0])
                equal = (rows == one) | (np.isnan(rows) & np.isnan(one))
                same &= equal.reshape(tied.size, -1).all(axis=1)
            if same.any():
                champion = int(tied[np.argmax(same)])
        pair = jnp.array([champion, champion])
        # With a padded compact axis the per-position arrays run past the real positions; the
        # record describes the substrate, so it keeps the real ones (the champion's evaluation
        # arguments keep the padded shapes the evaluation used).
        n_real = None if eval_positions is None else len(eval_positions)
        self._last_eval = {
            'forward_one': forward_one,
            'champion_args': tuple(None if a is None else a[pair] for a in genome_args),
            'champion_index': champion,
            'champion_fitness': float(fitnesses[champion]),
            'generation': self._current_generation,
            'champion_genome': tuple(np.asarray(part[champion]) for part in cppn_population[:2]),
            'mask_full': np.asarray(masks_full[champion]),
            'mask_eval': np.asarray(masks_eval[champion])[:n_real],
            'eval_positions': None if eval_positions is None else np.asarray(eval_positions),
            'act_indices': (None if act_indices is None
                            else np.asarray(act_indices[champion])[:n_real]),
            'agg_indices': (None if ih_agg_indices is None
                            else np.asarray(ih_agg_indices[champion])[:n_real]),
            **{name: weights[champion] for name, weights in (
                full_weights if full_weights is not None
                else {'W1_full': self._cached_W1, 'W2_full': self._cached_W2}).items()},
        }

    def _feedforward_core_blockers(self) -> List[str]:
        """Why this configuration's generation is not the plain feed-forward one of
        `_feedforward_generation`.

        Empty when the generation is HMR's loop's scope: grid substrate, a one-output CPPN, no
        hidden-to-hidden connections, LEO, neuromodulation, multi-task or streaming, activation and
        aggregation 'disabled' or 'global'. Each entry names a setting that needs the general path
        of run_generation_verbose.
        """
        blockers = []
        if self.network_model != 'grid':
            blockers.append(f"recurrence.network_model {self.network_model!r}")
        if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
            blockers.append("hidden-to-hidden connections (recurrence.allow_hidden_to_hidden)")
        if self._forward_mode != ForwardPassMode.DENSE_ONLY:
            blockers.append(f"the forward mode {self._forward_mode.name}")
        if self.enable_streaming:
            blockers.append("execution.mode 'streaming'")
        if self.use_leo:
            blockers.append("LEO (leo.enabled)")
        if self.num_cppn_outputs > 1:
            blockers.append(f"a CPPN with {self.num_cppn_outputs} outputs")
        if self.neuromod_config is not None:
            blockers.append("neuromodulation")
        if self.multitask_config is not None and getattr(self.multitask_config, 'enabled', False):
            blockers.append("multi-task evaluation (multitask.enabled)")
        if self.dynamic_functions_mode not in ('disabled', 'global'):
            blockers.append(f"dynamic_functions.mode {self.dynamic_functions_mode!r}")
        if self.agg_mode not in ('disabled', 'global'):
            blockers.append(f"aggregation.mode {self.agg_mode!r}")
        if self.hh_aggregation_mode == 'dynamic':
            blockers.append("aggregation.hh_aggregation_mode 'dynamic'")
        if self._df_sparsity_method != 'none' or self._df_scaling_method != 'none':
            blockers.append("dynamic_functions sparsity or scaling")
        return blockers

    def _substrate_query_rows(self) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
        """The CPPN input rows of the substrate's inputs and outputs, one array per coordinate."""
        positions = get_hierarchical_grid(self.max_depth).all_positions
        return (
            substrate_query_rows(
                self._cached_input_coords, positions, True, self.geometry_seeding_enabled
            ),
            substrate_query_rows(
                self._cached_output_coords, positions, False, self.geometry_seeding_enabled
            ),
        )

    def _query_population(self, state: Any, cppns_transformed: Tuple, rows: Sequence[jnp.ndarray],
                          device_id: int = 0) -> jnp.ndarray:
        """The standard generation's one-output CPPN queries (its query backend): (pop, sources,
        positions).

        One device, QUERY_POPULATION_CHUNK genomes per call, results kept on the device; under a
        split route's `DeviceSplit`, the same calls spread over its devices
        (`query_population_split`) and gathered on the first.
        """
        split = getattr(self, '_device_split', None)
        if split is not None and len(split.device_ids) > 1:
            return query_population_split(
                state, cppns_transformed, rows, self._jitted_cppn_forward, split
            )
        return query_population_on_device(state, cppns_transformed, rows, self._jitted_cppn_forward,
                                          device_id=device_id)

    def _eval_chunk_size(self, eval_length: int, pop_size: int) -> Optional[int]:
        """recurrence.eval_chunk_size, with -1 resolved for this evaluation axis and population."""
        eval_chunk_size = (
            getattr(self.extended_config, 'eval_chunk_size', None) if self.extended_config else None
        )
        if eval_chunk_size == -1:
            eval_chunk_size = _auto_eval_chunk_size(eval_length, pop_size)
        return eval_chunk_size

    def _evaluate_population(self, forward_one: Callable, genome_args: Tuple, chunk_by_scan: bool,
                             eval_chunk_size: Optional[int], inputs: jnp.ndarray,
                             targets: jnp.ndarray) -> jnp.ndarray:
        """The population's fitness under the generation's forward (the standard generation's eval
        backend).

        One device: one vmap over the population, or chunks of ``eval_chunk_size`` genomes, by a
        scan where the forward asks for one (see _build_forward).
        """
        split = getattr(self, '_device_split', None)
        if split is not None and len(split.device_ids) > 1:
            return evaluate_population_split(
                forward_one,
                genome_args,
                chunk_by_scan,
                eval_chunk_size,
                inputs,
                targets,
                self.fitness_mode,
                split,
            )
        return evaluate_population_on_device(
            forward_one,
            genome_args,
            chunk_by_scan,
            eval_chunk_size,
            inputs,
            targets,
            self.fitness_mode,
        )

    def _rescale_fitness(self, fitnesses: jnp.ndarray) -> jnp.ndarray:
        """1-MSE as R^2 = 1 - MSE/variance, floored at 0, when the problem's target variance asked
        for it.

        When target variance is small, 1-MSE fitness compresses the useful range; R^2 spreads the
        signal for evolutionary selection.
        """
        if self._fitness_rescale_variance is None:
            return fitnesses
        mse_values = 1.0 - fitnesses  # Recover MSE from 1-MSE fitness
        r2_values = 1.0 - mse_values / self._fitness_rescale_variance
        return jnp.maximum(0.0, r2_values)

    def _reached_per_level(self, masks: jnp.ndarray) -> List[int]:
        """Reached positions per level over the population (a host read; results untouched)."""
        h_grid = get_hierarchical_grid(self.max_depth)
        reach = np.asarray(masks)
        offsets = h_grid.level_offsets_static
        return [
            int(np.count_nonzero(reach[:, offsets[lv] : offsets[lv + 1]]))
            for lv in range(h_grid.num_levels)
        ]

    def _feedforward_generation(
        self,
        state: Any,
        query_rows: Tuple[Sequence[jnp.ndarray], Sequence[jnp.ndarray]],
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        step_times: Optional[Dict[str, Any]] = None,
        device_id: int = 0,
    ) -> FeedforwardGeneration:
        """One plain feed-forward generation, from ask to tell, as one traceable function.

        The steps run_generation_verbose takes for the configurations `_feedforward_core_blockers`
        passes: the opt-in pre-ask key split, ask and transform, the one-output CPPN queries
        (`_query_population`), masks and weights, the forward `_build_forward` gives and its
        evaluation (`_evaluate_population`), the optional rescaling, NaN to -inf, and tell. It reads
        the configuration and writes nothing to the engine: run_generation_verbose runs it op by op
        and keeps the records, counters and metrics itself, and execution.generation_loop
        'while_loop' compiles it inside lax.while_loop. Compiled, it is the same arithmetic up to
        rounding: XLA fuses the evaluation, which moves fitness by up to one float32 ulp.

        Args:
            state: Algorithm state.
            query_rows: (input rows, output rows), one array per coordinate
                (`_substrate_query_rows`).
            inputs: The problem's inputs.
            targets: The problem's targets.
            step_times: The generation's step timings, filled when given (never under tracing).
            device_id: The device of the queries.

        Returns:
            The generation's arrays and its forward.
        """
        def lap(name: str, start: float) -> float:
            if step_times is None:
                return start
            now = time.perf_counter()
            step_times[name] = now - start
            return now

        # Opt-in reproduction of HMR's pre-ask key split, as the general path does (see
        # create_config)
        if self._repro_extra_randkey_split:
            _, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        start = time.perf_counter() if step_times is not None else 0.0
        cppn_population = self._compiled_ask(state)
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        start = lap('step0_cppn_ask_transform', start)

        rows_in, rows_out = query_rows
        input_weights = self._query_population(state, cppns_transformed, rows_in, device_id)
        output_weights = self._query_population(state, cppns_transformed, rows_out, device_id)
        start = lap('step1_cppn_queries', start)

        h_grid = get_hierarchical_grid(self.max_depth)
        masks = substrate_masks(input_weights[:, 0, :], h_grid, self.variance_threshold,
                                initial_depth=self.initial_depth,
                                legacy_level0_variance=self._legacy_level0_variance)
        start = lap('step2_variance_masks', start)

        W1, W2 = substrate_weight_matrices(
            input_weights, output_weights, masks, self.max_weight,
            weight_threshold=self.weight_threshold,
            locality_radius=self.locality_radius,
            input_coords=self._cached_input_coords,
            output_coords=self._cached_output_coords,
            positions=h_grid.all_positions,
        )
        start = lap('step4_build_matrices', start)

        pop_size, _, total_positions = W1.shape
        ih_global_agg_fn = (
            AGGREGATION_FUNCTIONS.get(self.agg_global_function, AGGREGATION_FUNCTIONS['sum'])
            if self.agg_mode == 'global'
            else None
        )
        forward_one, genome_args, chunk_by_scan = self._build_forward(
            W1, W2, None, None, None, None, None, ih_global_agg_fn,
            self.dynamic_functions_mode != 'disabled', ih_global_agg_fn is not None,
            pop_size, total_positions, node_masks=masks,
        )
        fitnesses = self._evaluate_population(
            forward_one,
            genome_args,
            chunk_by_scan,
            self._eval_chunk_size(total_positions, pop_size),
            inputs,
            targets,
        )
        lap('step5_evaluation', start)

        # NaN (a NaN target) becomes -inf before selection sees it
        fitnesses, nan_count = sanitize_fitness_on_device(self._rescale_fitness(fitnesses))

        start = time.perf_counter() if step_times is not None else 0.0
        new_state = self._compiled_tell(*self._before_tell(state, fitnesses))
        lap('step6_neat_evolution', start)

        return FeedforwardGeneration(
            state=state, new_state=new_state, fitnesses=fitnesses, nan_count=nan_count,
            cppn_population=cppn_population, cppns_transformed=cppns_transformed, masks=masks,
            W1=W1, W2=W2, forward_one=forward_one, genome_args=genome_args,
        )

    def _run_generation_feedforward(
        self, state: Any, step_times: Dict[str, Any], total_start: float, device_id: int = 0
    ) -> Tuple[Any, AlgorithmMetrics]:
        """run_generation_verbose for the plain feed-forward configurations: the core op by op, then
        the same records the general path keeps for such a generation, then the shared tail."""
        # The palettes a runner may have set since the last generation are checked, as the general
        # path does
        self._check_function_palettes()
        gen = self._feedforward_generation(
            state,
            self._substrate_query_rows(),
            self._cached_inputs,
            self._cached_targets,
            step_times=step_times,
            device_id=device_id,
        )
        # No hidden-to-hidden discovery in this scope
        step_times['step3_phase2_discovery'] = 0.0

        self._cached_cppns_transformed = gen.cppns_transformed  # Cache for external eval (swarm)
        self._last_reached_per_level = self._reached_per_level(gen.masks)
        self._cppn_neuromodulation = None
        # The ceiling-sized arrays (no compaction without hidden-to-hidden connections)
        self._cached_W1 = gen.W1
        self._cached_W2 = gen.W2
        self._cached_masks_A = gen.masks
        self._cached_self_conn = None
        self._cached_sparse_hh = None
        self._cached_act_indices = None
        self._cached_hh_agg_indices = None
        self._cached_ih_agg_indices = None
        positions = gen.W1.shape[2]
        self._cached_fitnesses = gen.fitnesses

        return self._finish_generation(
            EvaluatedGeneration(
                state=gen.state,
                new_state=gen.new_state,
                fitnesses=gen.fitnesses,
                nan_count=int(gen.nan_count),
                cppn_population=gen.cppn_population,
                forward_one=gen.forward_one,
                genome_args=gen.genome_args,
                masks_full=gen.masks,
                masks_eval=gen.masks,
                eval_positions=None,
                act_indices=None,
                ih_agg_indices=None,
                W1=gen.W1,
                W2=gen.W2,
                sparse_hh=None,
                cache_hit=False,
                compact_pad=0,
                ceiling_positions=positions,
                total_positions=positions,
                num_levels=get_hierarchical_grid(self.max_depth).num_levels,
                pop_size=gen.W1.shape[0],
            ),
            step_times,
            total_start,
        )

    def _finish_generation(self, gen: EvaluatedGeneration, step_times: Dict[str, Any],
                           total_start: float) -> Tuple[Any, AlgorithmMetrics]:
        """The host side of a told generation, shared by every configuration: the champion record,
        HMR's live palette strategy, the counters, the metrics and palette evolution.

        Args:
            gen: The evaluated generation's arrays and the facts its metrics report.
            step_times: The generation's step timings.
            total_start: When the generation started (perf_counter).

        Returns:
            The state after tell (and after the live palette strategy) and the generation's metrics.
        """
        state, new_state, fitnesses, nan_fitness_count = (
            gen.state,
            gen.new_state,
            gen.fitnesses,
            gen.nan_count,
        )
        cppn_population, forward_one, genome_args = (
            gen.cppn_population,
            gen.forward_one,
            gen.genome_args,
        )
        masks_full, masks_A, eval_positions = gen.masks_full, gen.masks_eval, gen.eval_positions
        act_indices, ih_agg_indices = gen.act_indices, gen.ih_agg_indices
        W1, W2, sparse_hh, cache_hit = gen.W1, gen.W2, gen.sparse_hh, gen.cache_hit
        compact_pad, ceiling_positions = gen.compact_pad, gen.ceiling_positions
        total_positions, num_levels, pop_size = gen.total_positions, gen.num_levels, gen.pop_size

        self._record_evaluation(
            forward_one, genome_args, fitnesses, cppn_population,
            masks_full, masks_A, eval_positions, act_indices, ih_agg_indices, new_state,
        )

        # HMR's live palette strategy: after the tell, as in HMR, and after the record above, whose
        # tie resolution reads the next population's first genome unmutated
        if self._live_strategy is not None or self._palette_strategy is not None:
            start = time.perf_counter()
            new_state = self.update_palette_from_strategy(new_state, float(jnp.max(fitnesses)))
            step_times['step6b_palette_update'] = time.perf_counter() - start

        # Update generation counter
        self._current_generation += 1

        # Cache fitnesses for NAS substrate extraction
        self._last_fitnesses = fitnesses

        # Metrics
        total_time = time.perf_counter() - total_start

        # Compute fitness statistics (over the finite entries when NaN was sanitized)
        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            fitnesses, nan_fitness_count)
        max_fitness = best_fitness

        custom_metrics = {
            'total_positions': total_positions,
            'step_timings': step_times,
            'method': 'unified_extended',
            'recurrence_mode': self.get_recurrence_mode(),
            'multi_hop_algorithm': (
                self.extended_config.multi_hop_algorithm if self.extended_config else 'none'
            ),
            'iteration_level': self.extended_config.iteration_level if self.extended_config else 0,
            'cache_hit': cache_hit,
            'nan_fitness_count': nan_fitness_count,
        }

        if sparse_hh is not None:
            custom_metrics['total_hh_connections'] = int(jnp.sum(sparse_hh.num_valid))
            custom_metrics['avg_hh_connections'] = float(jnp.mean(sparse_hh.num_valid))
            # HMR's unified model's names: the population's h->h count, and whether the cache served
            # the generation
            custom_metrics['sparse_hh_count'] = custom_metrics['total_hh_connections']
            step_times['hh_cache_status'] = 'cached' if cache_hit else 'refresh'
            truncated = _last_discovery_stats.get('truncated_per_genome')
            if (
                not cache_hit
                and truncated is not None
                and truncated.shape[0] == sparse_hh.num_valid.shape[0]
            ):
                # Genomes over the max_sparse_conns cap (0 but under sparse_cap_policy
                # 'truncate': the other policies refuse or grow), the largest set, and the cap the
                # discovery used
                custom_metrics['hh_truncated_genomes'] = int(np.count_nonzero(truncated))
                custom_metrics['hh_max_uncapped'] = int(
                    _last_discovery_stats['uncapped_per_genome'].max()
                )
                custom_metrics['hh_max_sparse_conns'] = int(
                    _last_discovery_stats['max_sparse_conns']
                )

        # Compute substrate efficiency metrics
        active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
        active_positions_mean = float(jnp.mean(active_counts))
        active_positions_min = int(jnp.min(active_counts))
        active_positions_max = int(jnp.max(active_counts))
        # Against the substrate *ceiling*, not the compacted evaluation space. On the
        # recurrent path total_positions is the compact count, and dividing by it
        # would report utilization near 1.0 for a substrate that in fact occupies a
        # handful of the ceiling's positions: inverting the study's central finding.
        # Per-genome active counts are themselves unaffected by compaction, since a
        # position is only dropped when no genome reached it.
        position_utilization = (
            active_positions_mean / ceiling_positions if ceiling_positions > 0 else 0.0
        )

        # Add hidden node counts to custom_metrics for benchmark reporting
        custom_metrics['avg_hidden_nodes'] = active_positions_mean
        custom_metrics['min_hidden_nodes'] = active_positions_min
        custom_metrics['max_hidden_nodes'] = active_positions_max
        custom_metrics['position_utilization'] = position_utilization
        # HMR's keys, under the same names and meanings.
        custom_metrics['variance_threshold'] = self.variance_threshold
        custom_metrics['num_levels'] = num_levels

        # The evaluated population's species (state is the state before tell) and topology.
        num_species, species_sizes, species_fitness = species_summary(state.species, fitnesses)
        topology = topology_summary(
            W1, W2, masks_A, sparse_hh.num_valid if sparse_hh is not None else None)

        # Compute weight matrix density
        w1_total = W1.size
        w2_total = W2.size
        if compact_pad:
            # The virtual positions are zero columns of W1 and zero rows of W2; they are not
            # part of the substrate, so they stay out of the denominators.
            w1_total = W1.shape[0] * W1.shape[1] * (W1.shape[2] - compact_pad)
            w2_total = W2.shape[0] * (W2.shape[1] - compact_pad) * W2.shape[2]
        w1_nonzero = int(jnp.sum(W1 != 0))
        w2_nonzero = int(jnp.sum(W2 != 0))
        w1_density = w1_nonzero / w1_total if w1_total > 0 else 0.0
        w2_density = w2_nonzero / w2_total if w2_total > 0 else 0.0

        # Track cumulative cache hits
        cache_hit_count = getattr(self, '_cache_hit_count', 0)
        if cache_hit:
            cache_hit_count += 1
        self._cache_hit_count = cache_hit_count

        # Store extended metrics with all fields
        self._extended_metrics = EMRRecurrenceMetrics(
            # Connection statistics
            total_connections=custom_metrics.get('total_hh_connections', 0),
            hidden_to_hidden_connections=custom_metrics.get('total_hh_connections', 0),
            direct_hh_connections=custom_metrics.get('total_hh_connections', 0),  # Before multi-hop
            # Performance metrics
            fitness=best_fitness,
            activate_time_used=self.activate_time,
            # Fine-grained timing breakdown (ms)
            time_cppn_ask_ms=step_times['step0_cppn_ask_transform'] * 1000,
            time_cppn_query_ms=step_times['step1_cppn_queries'] * 1000,
            time_variance_ms=step_times['step2_variance_masks'] * 1000,
            time_hh_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            time_build_matrices_ms=step_times['step4_build_matrices'] * 1000,
            time_evaluation_ms=step_times['step5_evaluation'] * 1000,
            time_neat_evolution_ms=step_times['step6_neat_evolution'] * 1000,
            # Legacy timing (kept for compatibility)
            phase2_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            forward_pass_ms=step_times['step5_evaluation'] * 1000,
            # Multi-hop
            effective_iteration_level=(
                self.extended_config.iteration_level if self.extended_config else 0
            ),
            # Cache status
            cache_hit=cache_hit,
            cache_hit_count=cache_hit_count,
            cache_refresh_count=self._hh_cache._refresh_count if self._hh_cache else 0,
            # Substrate efficiency
            active_positions_mean=active_positions_mean,
            active_positions_min=active_positions_min,
            active_positions_max=active_positions_max,
            total_positions=total_positions,
            position_utilization=position_utilization,
            # Weight matrix density
            w1_density=w1_density,
            w2_density=w2_density,
            # Configuration echo
            variance_threshold_used=self.variance_threshold,
            band_threshold_used=self.band_threshold,
            max_weight_used=self.max_weight,
            # Population diversity
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
        )

        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=pop_size,
            time_elapsed=total_time,
            **topology,
        )
        # Add custom metrics
        metrics.custom_metrics = custom_metrics
        self._palette_evolution_step(fitnesses, custom_metrics)

        return new_state, metrics

    def run_generation_verbose(
        self,
        state: Any,
        problem: Any,
        skip_metrics: bool = False,
        device_id: int = 0,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with all extended features.

        This overrides the parent's run_generation_verbose to add:
        - Phase 2 discovery with multi-hop expansion
        - H->H caching
        - Extended metrics tracking

        Args:
            state: Current algorithm state
            problem: Problem instance
            skip_metrics: Accepted and not read: this path computes every metric (depth offload
                honors it)
            device_id: Device index for multi-GPU contexts (default 0)

        Returns:
            Tuple of (new_state, metrics)
        """
        step_times = {}
        total_start = time.perf_counter()

        # The plain feed-forward generation (HMR's loop's scope) runs through its traceable core,
        # the one execution.generation_loop 'while_loop' compiles.
        # Verbose runs keep the general path's diagnostics.
        if not self.verbose and not self._feedforward_core_blockers():
            return self._run_generation_feedforward(state, step_times, total_start, device_id)

        # Opt-in reproduction of HMR's pre-ask key split (default off; see create_config).
        # Mirrors the run_generation_verbose of HMR's dynamic-functions variant so that
        # extra_randkey_split=True reproduces the published per-seed results bit-for-bit.
        if getattr(self, '_repro_extra_randkey_split', False):
            _, randkey = jax.random.split(state.randkey)
            state = state.update(randkey=randkey)

        # Step 0: CPPN ask + transform
        start = time.perf_counter()
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        self._cached_cppns_transformed = cppns_transformed  # Cache for external eval (swarm)
        step_times['step0_cppn_ask_transform'] = time.perf_counter() - start

        if self.network_model == 'node_list':
            # The node-list model: its own discovery and forward, nothing of the grid
            return self._run_generation_node_list(state, cppn_population, cppns_transformed,
                                                  step_times, total_start, device_id)

        # Steps 1-4: Build W1/W2 (streaming or standard path)
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        # The CPPN's raw outputs for the input-to-node queries, (pop, inputs, positions, outputs),
        # kept on the standard path for HMR's reads (cppn_read 'input_mean', weight_source 'raw');
        # None on the streaming path.
        input_query_raw = None
        # The same for the node-to-output queries (the CPPN gate of neuromodulation.gate_source
        # 'cppn'), standard path
        output_query_raw = None

        if self.enable_streaming:
            # Streaming path: level-by-level CPPN queries for memory efficiency
            # One level's CPPN queries at a time (and the next level's input queries, for the
            # variance); W1/W2 stay full size
            if self.verbose:
                print(
                    f"[Streaming] Enabled for depth {self.max_depth} ({total_positions} positions)"
                )

            W1, W2, masks_A = self._build_matrices_streaming(
                state, cppns_transformed, h_grid, step_times
            )

            step_times['step1_cppn_queries'] = 0.0  # Included in streaming_build
            step_times['step2_variance_masks'] = 0.0  # Included in streaming_build
            step_times['step4_build_matrices'] = step_times.get('streaming_build', 0.0)
        else:
            # Standard path: query all positions at once (more memory)
            start = time.perf_counter()
            all_positions = h_grid.all_positions

            input_coords = self._cached_input_coords
            output_coords = self._cached_output_coords

            # CPPN Query: Standard or LEO-aware
            if self.use_leo:
                # LEO mode: Query CPPN for 2 outputs (weight + LEO expression)
                # Returns: (pop_size, num_inputs, total_positions, 2)
                input_all_weights_leo = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, input_coords, all_positions,
                    True, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                # Extract weight and LEO channels
                input_query_raw = input_all_weights_leo
                input_all_weights = input_all_weights_leo[:, :, :, 0]  # (pop, inputs, positions)
                # (pop, inputs, positions)
                input_leo_outputs = input_all_weights_leo[:, :, :, LEO_CHANNEL]

                # Output<-all positions: (pop_size, num_outputs, total_positions, 2)
                output_all_weights_leo = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, output_coords, all_positions,
                    False, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                output_query_raw = output_all_weights_leo
                output_all_weights = output_all_weights_leo[:, :, :, 0]  # (pop, outputs, positions)
                # (pop, outputs, positions)
                output_leo_outputs = output_all_weights_leo[:, :, :, LEO_CHANNEL]

                if self.verbose:
                    print("[LEO] Queried CPPN with 2 outputs (weight + LEO)")
            elif self.num_cppn_outputs > 1:
                # Multi-output CPPN (activation/aggregation have dedicated outputs)
                # Use the multi-output query path which preserves the output dimension
                # Returns: (pop_size, num_inputs, total_positions, num_cppn_outputs)
                input_all_multi = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, input_coords, all_positions,
                    True, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                # Extract weight channel (output 0)
                input_query_raw = input_all_multi
                input_all_weights = input_all_multi[:, :, :, 0]  # (pop, inputs, positions)

                output_all_multi = batch_query_population_multi_source_multi_output(
                    state, cppns_transformed, output_coords, all_positions,
                    False, self._jitted_cppn_forward,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                output_query_raw = output_all_multi
                output_all_weights = output_all_multi[:, :, :, 0]  # (pop, outputs, positions)

                if self.verbose:
                    print(f"[Multi-output CPPN] {self.num_cppn_outputs} outputs, "
                          f"indices={self._cppn_output_indices}")
            else:
                # Standard mode: Query CPPN for 1 output (weight only), through the query backend,
                # which keeps the outputs on the device (0.1.0 sent them to the host and back
                # through numpy)
                rows_in, rows_out = self._substrate_query_rows()
                # Input->all positions: (pop_size, num_inputs, total_positions)
                input_all_weights = self._query_population(
                    state, cppns_transformed, rows_in, device_id
                )
                input_query_raw = input_all_weights[..., None]

                # Output<-all positions: (pop_size, num_outputs, total_positions)
                output_all_weights = self._query_population(
                    state, cppns_transformed, rows_out, device_id
                )

            step_times['step1_cppn_queries'] = time.perf_counter() - start

            # Step 2: Variance + subdivision masks
            start = time.perf_counter()
            # Use input[0]->all_positions as variance source (2D: pop_size, total_positions)
            all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)
            masks_A = substrate_masks(
                all_weights_for_variance, h_grid, self.variance_threshold,
                initial_depth=self.initial_depth,
                legacy_level0_variance=self._legacy_level0_variance,
            )
            # Observable for substrate depth, mirroring the offload path: without
            # it there is no way to see which levels a mode actually reached.
            # Pure read; results untouched.
            self._last_reached_per_level = self._reached_per_level(masks_A)
            step_times['step2_variance_masks'] = time.perf_counter() - start

            # Step 4: Build W1/W2 matrices
            start = time.perf_counter()
            weight_thresh = 0.1  # Local constant (same as parent)
            max_weight = self.max_weight

            # Broadcast mask: (pop_size, 1, total_positions) for weight masking
            active_mask_broadcast = masks_A[:, None, :]

            # Weight Processing: Standard or LEO-aware
            if self.use_leo:
                # LEO mode: Use dual-threshold gating (weight + LEO)
                # Process W1 with LEO sparsification
                W1_leo_weights = jnp.tanh(input_all_weights)  # Normalize to [-1, 1]
                W1_leo_expressions = jnp.tanh(input_leo_outputs)  # Normalize LEO outputs

                # Apply locality penalty before LEO sparsification (if configured)
                if self.locality_radius is not None:
                    # Apply penalty to raw weights (before LEO gating)
                    W1_leo_weights = apply_locality_penalty_to_weights(
                        W1_leo_weights, input_coords, all_positions, self.locality_radius
                    )
                    if self.verbose:
                        print(f"[Locality+LEO] Applied penalty (r={self.locality_radius}) to W1")

                # Apply LEO sparsification per source (num_inputs)
                # vectorized_weight_sparsification_with_leo expects (N,) or (batch, N)
                # W1_leo_weights is (pop, inputs, positions) - reshape for batch processing
                pop_size_w = W1_leo_weights.shape[0]
                num_inputs_w = W1_leo_weights.shape[1]
                total_pos_w = W1_leo_weights.shape[2]

                # Reshape to (pop * inputs, positions), apply LEO, reshape back
                W1_flat = W1_leo_weights.reshape(pop_size_w * num_inputs_w, total_pos_w)
                W1_leo_flat = W1_leo_expressions.reshape(pop_size_w * num_inputs_w, total_pos_w)
                W1_sparsified = leo_sparsify(
                    W1_flat, W1_leo_flat, self.leo_mode,
                    weight_thresh, self.leo_threshold, max_weight,
                )
                W1_raw = W1_sparsified.reshape(pop_size_w, num_inputs_w, total_pos_w)

                # Apply active mask from variance subdivision
                W1 = jnp.where(active_mask_broadcast, W1_raw, 0.0)

                # Process W2 with LEO sparsification
                W2_leo_weights = jnp.tanh(output_all_weights)
                W2_leo_expressions = jnp.tanh(output_leo_outputs)

                if self.locality_radius is not None:
                    # Transpose temporarily for penalty
                    W2_for_penalty = W2_leo_weights.transpose(0, 2, 1)
                    W2_for_penalty = apply_locality_penalty_to_weights(
                        W2_for_penalty, all_positions, output_coords, self.locality_radius
                    )
                    W2_leo_weights = W2_for_penalty.transpose(0, 2, 1)
                    if self.verbose:
                        print(f"[Locality+LEO] Applied penalty (r={self.locality_radius}) to W2")

                num_outputs_w = W2_leo_weights.shape[1]
                W2_flat = W2_leo_weights.reshape(pop_size_w * num_outputs_w, total_pos_w)
                W2_leo_flat = W2_leo_expressions.reshape(pop_size_w * num_outputs_w, total_pos_w)
                W2_sparsified = leo_sparsify(
                    W2_flat, W2_leo_flat, self.leo_mode,
                    weight_thresh, self.leo_threshold, max_weight,
                )
                W2_raw = W2_sparsified.reshape(pop_size_w, num_outputs_w, total_pos_w)

                W2 = jnp.where(active_mask_broadcast, W2_raw, 0.0)
                W2 = W2.transpose(0, 2, 1)  # (pop, outputs, positions) -> (pop, positions, outputs)

                if self.verbose:
                    # Log LEO sparsification statistics
                    leo_valid_w1 = (W1 != 0).sum()
                    leo_valid_w2 = (W2 != 0).sum()
                    print(f"[LEO] W1 non-zero after LEO gating: {leo_valid_w1}")
                    print(f"[LEO] W2 non-zero after LEO gating: {leo_valid_w2}")
            else:
                # tanh scaling, optional locality penalty, then mask and threshold.
                # W1: (pop, num_inputs, total_positions); W2: (pop, total_positions, num_outputs)
                W1, W2 = substrate_weight_matrices(
                    input_all_weights, output_all_weights, masks_A, self.max_weight,
                    weight_threshold=self.weight_threshold,
                    locality_radius=self.locality_radius,
                    input_coords=input_coords,
                    output_coords=output_coords,
                    positions=all_positions,
                )
                if self.locality_radius is not None and self.verbose:
                    print(f"[Locality] Applied penalty (r={self.locality_radius}) to W1 and W2")

            step_times['step4_build_matrices'] = time.perf_counter() - start

        # Neuromodulation from dedicated CPPN outputs (HMR's Levels 1 to 3), read at the queries
        # that built the weight matrices: a gate per connection (static gating multiplies the
        # weights here, so every later reader sees the gated substrate; context gating keeps the
        # logits for its per-case gates), or the modulatory factor and node type the modulatory
        # options read, each averaged over the input queries as HMR read them (modulatory_cppn_read
        # 'input_mean'; the self read takes them from the self-connection query below).
        self._cppn_neuromodulation = None
        modulatory_reads = getattr(self, '_modulatory_cppn_reads', ())
        if self.neuromod_config is not None and 'gate' in self._cppn_output_indices:
            gate = self._cppn_output_indices['gate']
            active = masks_A[:, None, :]
            raw_in = input_query_raw[..., gate]  # (pop, inputs, positions)
            raw_out = output_query_raw[..., gate]  # (pop, outputs, positions)
            if self.neuromod_config.mode == 'static_gating':
                def scaled(raw):
                    return jnp.where(
                        active,
                        apply_gate_scaling(
                            raw,
                            self.neuromod_config.gate_scaling,
                            self.neuromod_config.gate_threshold,
                            self.neuromod_config.gate_hardness,
                        ),
                        0.0,
                    )
                G1, G2 = scaled(raw_in), jnp.transpose(scaled(raw_out), (0, 2, 1))
                self._cppn_neuromodulation = {
                    'G1': G1,
                    'G2': G2,
                    'W1_ungated': W1,
                    'W2_ungated': W2,
                }
                W1 = W1 * G1
                W2 = W2 * G2
            else:
                self._cppn_neuromodulation = {
                    'G1_logit': jnp.where(active, raw_in, 0.0),
                    'G2_logit': jnp.transpose(jnp.where(active, raw_out, 0.0), (0, 2, 1)),
                }
        elif (self.neuromod_config is not None and modulatory_reads
              and self.neuromod_config.modulatory_cppn_read == 'input_mean'):
            self._cppn_neuromodulation = {
                name: jnp.mean(input_query_raw[..., self._cppn_output_indices[name]], axis=1)
                for name in ('node_type', 'mod_factor') if name in modulatory_reads
            }

        # Step 3: Phase 2 discovery (with multi-hop)
        # This step runs AFTER matrices are built (needs masks_A)
        start = time.perf_counter()
        sparse_hh = None
        cache_hit = False

        if self.extended_config is not None and self.extended_config.allow_hidden_to_hidden:
            # Log config being used
            if self.verbose:
                print(f"[Gen {self._current_generation}] H->H config: "
                      f"iteration_level={self.extended_config.iteration_level}, "
                      f"cache_enabled={self.extended_config.hh_cache_enabled}")

            # Check cache
            if self._hh_cache is not None and not self._hh_cache.should_refresh(
                self._current_generation, masks_A
            ):
                # Cache hit
                sparse_hh = self._hh_cache.get_cached()
                cache_hit = True
                if self.verbose:
                    print(f"[Gen {self._current_generation}] Cache hit - reusing connections")
            else:
                # Need to discover
                if self.verbose:
                    print(f"[Gen {self._current_generation}] Cache miss - running discovery")
                sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                    state, cppns_transformed, h_grid, masks_A,
                    self.band_threshold, self.max_weight,
                    self.extended_config, self._jitted_cppn_forward,
                    verbose=self.verbose,
                    device_id=device_id,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                    use_leo=self.use_leo,
                    leo_threshold=self.leo_threshold, leo_mode=self.leo_mode,
                    num_cppn_outputs=self.num_cppn_outputs,
                )

                # Update cache
                if self._hh_cache is not None:
                    self._hh_cache.update_cache(sparse_hh, masks_A, self._current_generation)

        step_times['step3_phase2_discovery'] = time.perf_counter() - start

        # Cache weight matrices for multi-task evaluation
        # These are used by run_generation_multitask() to evaluate on multiple tasks
        # with different NT vectors without rebuilding W1/W2
        self._cached_W1 = W1
        self._cached_W2 = W2
        # The active positions of the same arrays (the per-genome orthogonality bonus masks the
        # hidden activity)
        self._cached_masks_A = masks_A

        # Query self-connections for per-node properties (activation/aggregation)
        # when using multi-output CPPNs with dedicated outputs
        self._cached_self_conn = None
        _need_self_conn = self.num_cppn_outputs > 1 and (
            self.dynamic_functions_mode == 'cppn_output' or self.agg_mode == 'cppn_output'
        )
        # Also query self-connections when the neuromod decode is configured to use
        # them as the receptor/base-gain source (the HMR/paper decode). Default off.
        _need_self_conn = _need_self_conn or (
            self.neuromod_config is not None
            and (
                getattr(self.neuromod_config, 'use_self_connection_query', False)
                or (
                    self.neuromod_config.mode == 'true_neuromodulation'
                    and not self.neuromod_config.receptor_from_weight
                )
                # Modulatory neurons reading their CPPN outputs at the self-connection
                or (bool(modulatory_reads) and self.neuromod_config.modulatory_cppn_read == 'self')
            )
        )
        if _need_self_conn:
            self_conn_positions = h_grid.all_positions
            self_conn_outputs = batch_query_population_self_connections(
                state, cppns_transformed, self_conn_positions,
                self._jitted_cppn_forward,
                num_cppn_outputs=self.num_cppn_outputs,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
                device_id=device_id,
            )
            self._cached_self_conn = self_conn_outputs
            if self.verbose:
                print(f"[CPPN Self-Conn] shape={self_conn_outputs.shape}, "
                      f"indices={self._cppn_output_indices}")

        # Cache sparse H->H connections for multi-task evaluation
        # This enables run_generation_multitask() to use H->H connections
        self._cached_sparse_hh = sparse_hh

        # Multi-task evaluation is also excluded from compaction.
        # run_generation_multitask calls this function to build the substrate and
        # then re-evaluates each task from self._cached_W1/_cached_W2, which are
        # cached *above* deliberately so they stay ceiling-sized. The per-position
        # neuromodulation state (the static and context gates) is derived *below*, so
        # compaction makes it compact while the cached weights stay full, and the
        # two do not line up. That combination would fail 165 tests across the
        # multi-task, two-module and per-task-activation paths, every one with
        # "Incompatible shapes for broadcasting" at generation 0.
        #
        # Caching the compact matrices instead would make multi-task self-consistent
        # but would change what every other consumer of _cached_W1 receives.
        _neuromod_mode = getattr(self.neuromod_config, 'mode', None) if self.neuromod_config else None
        _multitask_on = (self.multitask_config is not None
                         and getattr(self.multitask_config, 'enabled', False))
        _compaction_blocked = _neuromod_mode == 'modulatory_neurons' or _multitask_on

        # In-generation view of the self-connection query. Compaction narrows
        # this local; self._cached_self_conn stays ceiling-sized for multi-task.
        self_conn_eval = self._cached_self_conn
        ceiling_positions = W1.shape[2] if W1.ndim > 2 else W1.shape[1]
        # The ceiling-sized mask, and the ceiling position of every evaluated position (None:
        # the identity), for evaluate_on_data's substrate statistics.
        masks_full = masks_A
        eval_positions = None
        # Virtual positions appended to the compact axis (execution.compaction_bucket 'auto');
        # 0 without compaction or with 'exact'. Counts and records use the real positions only.
        compact_pad = 0
        if self._forward_mode != ForwardPassMode.DENSE_ONLY and not _compaction_blocked:
            from emr_hyperneat.execution.compaction import (
                bucket_active_set,
                compact_weights,
                find_active_positions,
                pad_positions,
                remap_sparse_hh,
            )

            _active = find_active_positions(masks_A)
            # The random per-node modes keep exact shapes: random_generation draws its indices with
            # the position count as the shape, and a longer draw is not the same draw; random_fixed
            # gathers its whole-axis draw at exactly the active positions.
            _random_draws = (self.dynamic_functions_mode in ('random_fixed', 'random_generation')
                             or getattr(self, 'agg_mode', None) in ('random_fixed', 'random_generation'))
            if _active.num_active and _active.num_active < _active.total_positions:
                # Bucketed: the compact axis is padded to a bucketed length so its shape repeats
                # across generations. A count that lands exactly on a bucket length needs no padding
                # but still takes the host path: there the exact path's device gathers, and the later
                # ops on the gathered mask, compiled a separate set of executables (13 on a GPU) the
                # first time such a count occurred.
                bucketed = self.execution_config.compaction_bucket == 'auto' and not _random_draws
                if bucketed:
                    _active = bucket_active_set(_active)
                    compact_pad = _active.length - _active.num_active
                # W2 is (pop, positions, n_out) here, unlike the offload path's host
                # layout, so it is moved to positions-last for the shared helper and
                # back afterwards.
                W1, _W2_last = compact_weights(
                    W1, jnp.transpose(W2, (0, 2, 1)), _active
                )
                W2 = jnp.transpose(_W2_last, (0, 2, 1))
                if bucketed:
                    masks_A = pad_positions(masks_A, _active, axis=1, fill=False)
                else:
                    masks_A = masks_A[:, _active.indices]
                eval_positions = _active.indices
                sparse_hh = remap_sparse_hh(sparse_hh, _active)
                # The self-connection query above ran over h_grid.all_positions, so
                # it is still ceiling-sized while everything else is compact, and
                # the `cppn_output` modes read act/agg indices straight off it.
                #
                # Compacted into a local, never into self._cached_self_conn. That
                # distinction is load-bearing: multi-task evaluation consumes the
                # cached self-connections alongside the uncompacted _cached_W1, so
                # shrinking the cache in place mismatches them. Doing exactly that
                # fails 165 tests across the multi-task and two-module paths, all
                # with "Incompatible shapes for broadcasting" at generation 0.
                # Position is axis 1 here: (pop, positions, cppn_outputs).
                if self_conn_eval is not None:
                    if bucketed:
                        # Finite raw outputs for the virtual nodes: they decode to an ordinary
                        # palette entry, finite at a zero pre-activation, and feed nothing.
                        self_conn_eval = pad_positions(self_conn_eval, _active, axis=1, fill=0.0)
                    else:
                        self_conn_eval = self_conn_eval[:, _active.indices, :]
                if self.verbose:
                    print(f"[Compact] recurrent evaluation on {_active.num_active:,} "
                          f"active of {_active.total_positions:,} positions "
                          f"({_active.reduction:,.0f}x reduction)"
                          + (f", padded to {_active.length:,}" if compact_pad else ""))

        # Log weight matrix statistics for feedforward investigation
        if self.verbose and self._forward_mode == ForwardPassMode.DENSE_ONLY:
            active_counts = masks_A.sum(axis=1)  # Per-genome active position counts
            W1_nonzero = (W1 != 0).sum(axis=(1, 2))  # Per-genome non-zero W1 entries
            W2_nonzero = (W2 != 0).sum(axis=(1, 2))  # Per-genome non-zero W2 entries
            print(f"[Feedforward Diag] Active positions: mean={float(active_counts.mean()):.1f}, "
                  f"min={int(active_counts.min())}, max={int(active_counts.max())}")
            print(f"[Feedforward Diag] W1 non-zero: mean={float(W1_nonzero.mean()):.1f}, "
                  f"W2 non-zero: mean={float(W2_nonzero.mean()):.1f}")
            print(f"[Feedforward Diag] W1 stats: min={float(W1.min()):.3f}, max={float(W1.max()):.3f}, "
                  f"mean={float(W1.mean()):.4f}")
            print(f"[Feedforward Diag] W2 stats: min={float(W2.min()):.3f}, max={float(W2.max()):.3f}, "
                  f"mean={float(W2.mean()):.4f}")
            print(f"[Feedforward Diag] Total positions: {total_positions}, "
                  f"Variance threshold: {self.variance_threshold}")

        # Step 5: Evaluation
        start = time.perf_counter()
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        pop_size = W1.shape[0]
        # eval_length is the position axis of the arrays, padding included (the chunking heuristics
        # key on it: it is the compiled shape); total_positions counts the real positions, for the reports.
        eval_length = W1.shape[2] if len(W1.shape) > 2 else W1.shape[1]
        total_positions = eval_length - compact_pad

        # Chunk size from config (None or 0 = no chunking, -1 = auto, the same heuristics as CPPN
        # queries)
        eval_chunk_size = self._eval_chunk_size(eval_length, pop_size)

        # Dynamic functions: compute per-node activation indices based on mode
        act_indices = None  # Per-node activation indices (num_positions,)
        use_dynamic_eval = self.dynamic_functions_mode not in ('disabled',)

        if self.dynamic_functions_mode == 'global':
            # Global mode: all nodes use same activation (no per-node indices needed)
            # The hidden_act_fn is already set in _parse_dynamic_functions_config
            act_indices = None
        elif (self.dynamic_functions_mode == 'weight_interpretation'
              or (self.dynamic_functions_mode == 'modular'
                  and self.df_modular_config['activation_method'] != 'none')):
            # Derive activation indices from W1 weight patterns (W1: (pop, num_inputs,
            # num_positions)), with the interpretation or modular's activation method (modular
            # 'none' is tanh everywhere, _hidden_act_fn). With a palette the decode picks a palette
            # position and maps it to the function's ACTIVATION_LIST index, as the cppn_output and
            # random modes do.
            method = (
                self.df_interpretation
                if self.dynamic_functions_mode == 'weight_interpretation'
                else self.df_modular_config['activation_method']
            )
            normalized = activation_interpretation_signal(
                W1 if self.df_weight_source == 'w1' else input_query_raw[..., 0], method)
            num_options = (
                len(self.df_palette) if self.df_palette is not None else self.df_num_activations
            )
            act_indices = continuous_to_index(normalized, num_options, self.df_index_decode)
            if self.df_palette is not None:
                act_indices = self.df_palette[act_indices]
        elif self.dynamic_functions_mode in ('random_fixed', 'random_generation'):
            # Random per-node activation (a negative control): drawn at the first generation and
            # kept (random_fixed, per population slot and grid position) or drawn every generation
            # (random_generation); see random_function_indices for the two scopes.
            num_activations = (len(self.df_palette) if self.df_palette is not None
                               else self.df_num_activations)
            if self.dynamic_functions_mode == 'random_fixed':
                # Drawn over the whole position axis and gathered at the positions evaluated, so
                # a position keeps its function while the compacted active set changes. A draw the
                # length of generation 0's compacted axis, reused whole, would fail a recurrent run
                # ("Incompatible shapes for broadcasting") as soon as the active count changed. A
                # feed-forward run evaluates the whole axis, so the gather changes nothing there.
                if self._random_fixed_indices is None:
                    self._random_fixed_indices = random_function_indices(
                        self._random_scope, self._random_mode_seed, state.randkey, 9997, None,
                        num_activations, self.df_palette, pop_size, ceiling_positions)
                act_indices = (self._random_fixed_indices if eval_positions is None
                               else self._random_fixed_indices[:, eval_positions])
            else:
                act_indices = random_function_indices(
                    self._random_scope, self._random_mode_seed, state.randkey, 9999,
                    self._current_generation, num_activations, self.df_palette, pop_size,
                    total_positions)
        elif self.dynamic_functions_mode == 'cppn_output':
            # CPPN output mode: activation index from the dedicated CPPN output at each
            # node's self-connection, or from W1 for a one-output CPPN
            decode_palette, num_activations = self._activation_decode_palette()
            if self.df_cppn_read == 'input_mean':
                # HMR's read: the channel averaged over the input-to-node queries (unmasked)
                act_indices = continuous_to_index(
                    jnp.mean(input_query_raw[..., self._cppn_output_indices['activation']], axis=1),
                    num_activations, self.df_index_decode)
                if decode_palette is not None:
                    act_indices = decode_palette[act_indices]
            else:
                act_indices = cppn_output_function_indices(
                    self_conn_eval, masks_A, W1,
                    channel=self._cppn_output_indices['activation'],
                    num_options=num_activations,
                    num_cppn_outputs=self.num_cppn_outputs,
                    palette=decode_palette,
                    decode=self.df_index_decode,
                )

        # The palettes a runner may have set since the last generation (df_palette, agg_palette) are
        # checked here (0.1.0 failed with an IndexError on an entry beyond its function list and
        # wrapped a negative one).
        self._check_function_palettes()
        # With an aggregation palette every decode below picks a palette position and maps it to its
        # AGGREGATION_LIST index (to_agg); without one the positions are
        # AGGREGATION_LIST[:agg_num_aggregations].
        agg_palette = self.agg_palette
        agg_options = len(agg_palette) if agg_palette is not None else self.agg_num_aggregations

        def to_agg(indices):
            return indices if agg_palette is None or indices is None else agg_palette[indices]

        # The weights the aggregation interpretations read (aggregation.weight_source)
        agg_weights = W1 if self.agg_weight_source == 'w1' else input_query_raw[..., 0]

        # Step 4b: Compute H->H aggregation indices if dynamic mode enabled
        hh_agg_indices = None
        if self.hh_aggregation_mode == 'dynamic':
            if self.num_cppn_outputs > 1 and self_conn_eval is not None and self.agg_mode == 'cppn_output':
                # Use dedicated CPPN output for H->H aggregation (multi-output CPPN)
                agg_output_idx = self._cppn_output_indices['aggregation']
                agg_raw = self_conn_eval[:, :, agg_output_idx]  # (pop, positions)
                hh_agg_indices = to_agg(
                    continuous_to_index(agg_raw, agg_options, self.agg_index_decode)
                )
            else:
                # Fallback: the input side's weight interpretation.
                hh_agg_indices = to_agg(aggregation_from_weight_interpretation(
                    agg_weights, self.agg_interpretation, agg_options, self.agg_index_decode))

        # Compute I->H aggregation indices based on aggregation mode
        ih_agg_indices = None
        ih_global_agg_fn = None
        if self.agg_mode == 'global':
            ih_global_agg_fn = AGGREGATION_FUNCTIONS.get(
                self.agg_global_function, AGGREGATION_FUNCTIONS['sum']
            )
        elif self.agg_mode == 'cppn_output':
            # Dedicated CPPN output at the self-connection, or W1 for a one-output CPPN
            if self.agg_cppn_read == 'input_mean':
                # HMR's read: the channel averaged over the input-to-node queries (unmasked)
                ih_agg_indices = to_agg(
                    continuous_to_index(
                        jnp.mean(
                            input_query_raw[..., self._cppn_output_indices['aggregation']], axis=1
                        ),
                        agg_options,
                        self.agg_index_decode,
                    )
                )
            else:
                ih_agg_indices = cppn_output_function_indices(
                    self_conn_eval, masks_A, W1,
                    channel=self._cppn_output_indices['aggregation'],
                    num_options=agg_options,
                    num_cppn_outputs=self.num_cppn_outputs,
                    palette=agg_palette,
                    decode=self.agg_index_decode,
                )
        elif self.agg_mode == 'weight_interpretation':
            ih_agg_indices = to_agg(aggregation_from_weight_interpretation(
                agg_weights, self.agg_interpretation, agg_options, self.agg_index_decode
            ))
        elif self.agg_mode == 'modular':
            # dynamic_functions.modular.aggregation_method, decoded as weight_interpretation does;
            # 'none' is sum
            if self.df_modular_config['aggregation_method'] != 'none':
                ih_agg_indices = to_agg(
                    aggregation_from_weight_interpretation(
                        agg_weights,
                        self.df_modular_config['aggregation_method'],
                        agg_options,
                        self.agg_index_decode,
                    )
                )
        elif self.agg_mode in ('random_fixed', 'random_generation'):
            # Random per-node aggregation, as the activation's random modes (HMR's salts 7777,
            # 8888); random_fixed draws over the whole position axis, as the activation's does
            if self.agg_mode == 'random_fixed':
                if self._random_fixed_agg_indices is None:
                    self._random_fixed_agg_indices = random_function_indices(
                        self._random_agg_scope,
                        self._random_agg_seed,
                        state.randkey,
                        7777,
                        None,
                        agg_options,
                        agg_palette,
                        pop_size,
                        ceiling_positions,
                    )
                ih_agg_indices = (self._random_fixed_agg_indices if eval_positions is None
                                  else self._random_fixed_agg_indices[:, eval_positions])
            else:
                ih_agg_indices = random_function_indices(
                    self._random_agg_scope,
                    self._random_agg_seed,
                    state.randkey,
                    8888,
                    self._current_generation,
                    agg_options,
                    agg_palette,
                    pop_size,
                    total_positions,
                )

        use_ih_aggregation = (ih_agg_indices is not None or ih_global_agg_fn is not None)

        # Step 4d: HMR's sparsity and scaling (weight_interpretation's three mechanisms, modular's
        # layers 2 and 3), from the same unscaled W1 the indices above came from. Sparsity is a
        # per-node quantile (pop, positions), or the winner-take-all fraction, applied after the
        # activation by the feed-forward evaluation; scaling multiplies each node's incoming
        # weights.
        sparsity_quantiles = None
        wta_k_percent = None
        if self._df_sparsity_method == 'hierarchical':
            levels = level_sparsity_quantiles(
                get_hierarchical_grid(self.max_depth).level_offsets_static, self._df_sparsity_params
            )
            if levels.shape[0] != total_positions:
                # The feed-forward evaluation axis is the whole grid in grid order (compaction is
                # recurrent-only)
                raise RuntimeError(
                    f"hierarchical sparsity needs the grid's {levels.shape[0]} positions on the "
                    f"evaluation axis, which has {total_positions}"
                )
            sparsity_quantiles = jnp.broadcast_to(levels, (pop_size, total_positions))
        elif self._df_sparsity_method == 'threshold':
            sparsity_quantiles = node_sparsity_quantiles(
                W1 if self.df_weight_source == 'w1' else input_query_raw[..., 0])
        elif self._df_sparsity_method == 'wta':
            wta_k_percent = float(self._df_sparsity_params['wta_k_percent'])
        if self._df_scaling_method == 'stp_inspired':
            W1 = (
                W1
                * stp_depression(W1 if self.df_weight_source == 'w1' else input_query_raw[..., 0])[
                    :, None, :
                ]
            )

        # Cache per-node function indices for NAS gradient training extraction
        self._cached_act_indices = act_indices
        self._cached_hh_agg_indices = hh_agg_indices
        self._cached_ih_agg_indices = ih_agg_indices

        # Neuromodulation: derive receptor densities, base gains, and gates
        # Storage for neuromodulation data (set to None if disabled)
        neuromod_data = None
        if self.neuromod_config and self.neuromod_config.mode != 'disabled':
            neuromod_mode = self.neuromod_config.mode
            neuromod_data = {'mode': neuromod_mode}

            if neuromod_mode == 'static_gating' and self.neuromod_config.gate_source == 'cppn':
                # Level 1 from a CPPN output: the connections were gated where the weights were
                # built; no node gate
                static_gates = jnp.ones((pop_size, eval_length))
                neuromod_data['static_gates'] = static_gates

            elif neuromod_mode == 'context_gating' and self.neuromod_config.gate_source == 'cppn':
                # Level 2 from a CPPN output (HMR's): per-case gates sigmoid(logit + influence *
                # mean(input statistics))
                neuromod_data['gate_source'] = 'cppn'
                neuromod_data['G1_logit'] = self._cppn_neuromodulation['G1_logit']
                neuromod_data['G2_logit'] = self._cppn_neuromodulation['G2_logit']
                neuromod_data['context_influence'] = self.neuromod_config.context_influence

            elif neuromod_mode == 'static_gating':
                # Level 1: Static gating from weight interpretation
                # Gates are derived from mean absolute weight magnitude
                # Higher magnitude weights -> more active connections
                mean_abs_w = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                gate_raw = jnp.tanh(mean_abs_w * 2)  # In [0, 1): mean_abs_w is not negative
                static_gates = apply_gate_scaling(
                    gate_raw,
                    self.neuromod_config.gate_scaling,
                    self.neuromod_config.gate_threshold,
                    self.neuromod_config.gate_hardness,
                )  # (pop, num_positions)
                neuromod_data['static_gates'] = static_gates

            elif neuromod_mode == 'context_gating':
                # Level 2: Context-dependent gating (XdG-style)
                # Gate values depend on task context
                context_dim = self.neuromod_config.context_dim
                context_influence = self.neuromod_config.context_influence

                # Derive base gates from weights (same as static gating)
                mean_abs_w = jnp.mean(jnp.abs(W1), axis=1)  # (pop, num_positions)
                base_gates = jnp.tanh(mean_abs_w * 2)

                # Derive per-context gate modulation from W1 patterns
                # Use variance across inputs to create context-specific modulation
                weight_var = jnp.var(W1, axis=1)  # (pop, num_positions)
                context_modulation = jnp.zeros((pop_size, context_dim, eval_length))

                # Create context-specific modulations using different weight patterns
                for ctx_idx in range(context_dim):
                    # Rotate/phase-shift the weight patterns for different contexts
                    phase_shift = ctx_idx * (2 * jnp.pi / context_dim)
                    ctx_pattern = jnp.sin(weight_var * 5 + phase_shift)
                    context_modulation = context_modulation.at[:, ctx_idx, :].set(ctx_pattern)

                neuromod_data['base_gates'] = base_gates
                neuromod_data['context_modulation'] = context_modulation
                neuromod_data['context_influence'] = context_influence

            elif neuromod_mode == 'modulatory_neurons':
                # Level 3, modulatory neurons: activity modulation, some of each genome's own active
                # positions set a per-case gain for the others (Soltoggio's modulatory neurons gate
                # plasticity, which EMR does not have). The options choose the neurons, their count,
                # the signal, the per-node weights, the scope and a matched-gain control;
                # modulatory_source's bundle sets each option left unset.
                cfg = self.neuromod_config
                if cfg.modulatory_matched_gain != 'none' and self._cached_inputs.shape[0] < 2:
                    raise ValueError(
                        f"neuromodulation.modulatory_matched_gain {cfg.modulatory_matched_gain!r} "
                        "acts over the evaluation's cases and this problem has "
                        f"{self._cached_inputs.shape[0]}: with one case it equals 'none' (the mean "
                        "of one signal is that signal, a shift by one over one case is the "
                        "identity)"
                    )
                if modulatory_reads and cfg.modulatory_cppn_read == 'self':
                    # The outputs at each position's self-connection query (compaction is blocked
                    # for this mode, so the query covers the evaluated positions)
                    self._cppn_neuromodulation = {
                        name: self_conn_eval[:, :, self._cppn_output_indices[name]]
                        for name in ('node_type', 'mod_factor') if name in modulatory_reads
                    }

                # Each genome's modulatory neurons, counted over its own active positions: those of
                # highest input-weight variance or node-type output
                weight_var = (
                    jnp.var(W1, axis=1)  # (pop, num_positions)
                    if 'variance' in (cfg.modulatory_selection, cfg.modulatory_node_weights)
                    else None
                )
                score = (weight_var if cfg.modulatory_selection == 'variance'
                         else self._cppn_neuromodulation['node_type'])
                if cfg.modulatory_count_rule == 'floor':
                    mod_mask = modulatory_neuron_masks(score, masks_A, cfg.mod_neuron_ratio)
                elif cfg.modulatory_count_rule == 'ceil':
                    mod_mask = cppn_modulatory_masks(score, masks_A, cfg.mod_neuron_ratio)
                else:
                    mod_mask = modulatory_masks(score, masks_A, 'fixed', count=cfg.mod_neuron_count)

                neuromod_data['mod_mask'] = mod_mask
                neuromod_data['mod_connection_type'] = cfg.mod_connection_type
                neuromod_data['mod_decay'] = cfg.mod_decay
                neuromod_data['modulatory_options'] = (
                    cfg.modulatory_signal,
                    cfg.modulatory_node_weights,
                    cfg.modulatory_scope,
                    cfg.modulatory_matched_gain,
                )
                if cfg.modulatory_node_weights == 'variance':
                    # Each node's weight on the signal, from the variance of its input weights
                    mod_connection_weights = jnp.tanh(weight_var) * cfg.mod_decay
                    neuromod_data['mod_connection_weights'] = mod_connection_weights
                    self._mod_connection_weights = mod_connection_weights
                if cfg.modulatory_signal == 'sum_factor':
                    # A modulatory neuron's factor is tanh of the mod-factor output
                    neuromod_data['mod_factor'] = jnp.tanh(self._cppn_neuromodulation['mod_factor'])
                self._mod_neuron_mask = mod_mask

            elif neuromod_mode == 'true_neuromodulation':
                # Level 4: TRUE neuromodulation
                # NT vectors times receptor densities -> gain modulation
                num_nt = self.neuromod_config.num_nt_types
                mod_strength = self.neuromod_config.modulation_strength
                receptor_derivation = self.neuromod_config.receptor_derivation

                if self.neuromod_config.receptor_from_weight:
                    # Option A: Derive receptor densities from weights
                    # W1 shape: (pop, num_inputs, num_positions)
                    #
                    # Use first input's weights (not mean) to match original's single-value-per-position approach.
                    # The original uses CPPN output at self-connection coordinates (x,y,x,y,bias),
                    # which gives one value per position. Using first input's weights approximates this
                    # by giving one value per position rather than averaging multiple values.
                    #
                    # This is important for receptor_densities because:
                    # - mean() smooths out the values, losing the variation needed for diverse modulation
                    # - first input gives a consistent per-position property similar to self-connection query
                    # Choose the per-position weight source for receptor/base-gain.
                    if (getattr(self.neuromod_config, 'use_self_connection_query', False)
                            and self_conn_eval is not None):
                        # Exact HMR/paper reproduction: raw CPPN self-connection query at
                        # (x,y,x,y,bias); index 0 = weight output, the same source HMR uses
                        # (weight_values = self_conn_outputs[:, :, 0]). Already bounded ~[-1, 1].
                        weight_normalized = self_conn_eval[:, :, 0]
                    else:
                        # Default EMR approximation: first input's weight, normalized to ~[-1, 1].
                        # W1 = tanh(cppn_output) * max_weight, so W1/max_weight ~= tanh(cppn_output).
                        weight_for_receptor = W1[:, 0, :]  # (pop, num_positions) - use first input
                        weight_normalized = weight_for_receptor / self.max_weight

                    receptor_densities = derive_receptor_from_weight(
                        weight=weight_normalized, method=receptor_derivation, num_nt_types=num_nt
                    )  # (pop, num_positions, num_nt)

                    # Convert to [-1, 1] range to match Option B (enables both excitatory and inhibitory)
                    # This matches HMR's neuromodulation variant
                    receptor_densities = receptor_densities * 2.0 - 1.0

                    # Base gains: derived from weight magnitude, range [0.5, 1.0]
                    # Use normalized weight (already in [-1, 1]) to match original
                    base_gains = jnp.abs(weight_normalized) * 0.5 + 0.5
                else:
                    # Option B (HMR's formulas): each NT type's receptor density is a dedicated CPPN
                    # output at the node's self-connection, tanh-bounded; the base gain comes from
                    # the weight output there. HMR's clip to +-10 and its NaN guards are kept (the
                    # clip does nothing on the CPPN's tanh-bounded outputs). With four NT types the
                    # rule reads three receptors (the fourth NT channel inverts the outputs), so the
                    # fourth output has no effect, as in HMR.
                    base = self._cppn_output_indices['receptor']
                    receptor_densities = jnp.tanh(
                        jnp.clip(self_conn_eval[:, :, base : base + num_nt], -10.0, 10.0)
                    )
                    receptor_densities = jnp.where(
                        jnp.isnan(receptor_densities), 0.0, receptor_densities
                    )
                    base_gains = jnp.abs(jnp.clip(self_conn_eval[:, :, 0], -10.0, 10.0)) * 0.5 + 0.5
                    base_gains = jnp.where(jnp.isnan(base_gains), 0.75, base_gains)

                # Store for use in evaluation
                neuromod_data['receptor_densities'] = receptor_densities
                neuromod_data['base_gains'] = base_gains
                neuromod_data['num_nt'] = num_nt
                neuromod_data['mod_strength'] = mod_strength
                neuromod_data['modulation_mode'] = self.neuromod_config.modulation_mode
                neuromod_data['use_output_inversion'] = self.neuromod_config.use_output_inversion
                neuromod_data['node_rule'] = self.neuromod_config.node_rule
                neuromod_data['branch_gating'] = self.neuromod_config.branch_gating

                # Get default NT vector for single-task evaluation
                if self.multitask_config and self._multitask_state.get('nt_vectors') is not None:
                    # Use first task's NT vector by default
                    neuromod_data['neurotransmitter'] = self._multitask_state['nt_vectors'][0]
                else:
                    # Default NT vector: [0.5, 0.5, 0.5, 1.0]
                    neuromod_data['neurotransmitter'] = jnp.array(
                        [0.5] * min(3, num_nt) + [1.0] if num_nt >= 4 else [0.5] * num_nt
                    )

                # Store in class attribute
                self._neuromod_true['receptor_densities'] = receptor_densities
                self._neuromod_true['base_gains'] = base_gains

        # One per-genome forward for this configuration: the population's fitness here and
        # evaluate_on_data's held-out score both come from it (see _build_forward).
        recurrent = (sparse_hh is not None
                     and self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH)
        forward_one, genome_args, chunk_by_scan = self._build_forward(
            W1, W2, sparse_hh if recurrent else None, neuromod_data, act_indices,
            hh_agg_indices, ih_agg_indices, ih_global_agg_fn, use_dynamic_eval,
            use_ih_aggregation, pop_size, eval_length,
            node_masks=masks_A, sparsity_quantiles=sparsity_quantiles, wta_k_percent=wta_k_percent,
        )
        fitnesses = self._evaluate_population(
            forward_one, genome_args, chunk_by_scan, eval_chunk_size, inputs_batch, targets_batch
        )

        step_times['step5_evaluation'] = time.perf_counter() - start

        # STEP 5b: Fitness Rescaling (optional)
        fitnesses = self._rescale_fitness(fitnesses)

        # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees it.
        fitnesses, nan_fitness_count = sanitize_fitness(fitnesses)

        # Cache per-individual fitness for external access (e.g., Baldwin Effect)
        self._cached_fitnesses = fitnesses

        # Step 6: NEAT evolution
        start = time.perf_counter()
        new_state = self._compiled_tell(*self._before_tell(state, fitnesses))
        step_times['step6_neat_evolution'] = time.perf_counter() - start

        return self._finish_generation(
            EvaluatedGeneration(
                state=state,
                new_state=new_state,
                fitnesses=fitnesses,
                nan_count=nan_fitness_count,
                cppn_population=cppn_population,
                forward_one=forward_one,
                genome_args=genome_args,
                masks_full=masks_full,
                masks_eval=masks_A,
                eval_positions=eval_positions,
                act_indices=act_indices,
                ih_agg_indices=ih_agg_indices,
                W1=W1,
                W2=W2,
                sparse_hh=sparse_hh,
                cache_hit=cache_hit,
                compact_pad=compact_pad,
                ceiling_positions=ceiling_positions,
                total_positions=total_positions,
                num_levels=h_grid.num_levels,
                pop_size=pop_size,
            ),
            step_times,
            total_start,
        )
