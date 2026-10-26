"""Multi-task and neuromodulated evaluation: the evaluator factories, the multi-task generation, and the
methods that evaluate with given neurotransmitters."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat._compat.core.base_algorithm import AlgorithmMetrics
from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid
from emr_hyperneat.engine.config import ForwardPassMode
from emr_hyperneat.engine.functions import ACTIVATION_FUNCTIONS
from emr_hyperneat.engine.metrics import fitness_statistics, sanitize_fitness, species_summary
from emr_hyperneat.engine.multitask import (
    BRANCH_GATING_MODES,
    GENERALIST_BONUS_TYPES,
    MULTITASK_FITNESS_MODES,
    MultiTaskConfig,
    MultiTaskMetrics,
    ORTHOGONALITY_METRICS,
    _MULTITASK_EVALUATORS,
    _refuse_per_node_in_multitask,
    check_branch_gating_and_penalty,
    compute_fitness,
    compute_modulation_with_branch_gating,
    compute_population_orthogonality,
    refuse_anti_branch_gating,
)
from emr_hyperneat.engine.neuromodulation import (
    NT_TASK_PRESETS,
    get_nt_for_task,
    invert_outputs,
    true_neuromodulation_node_rule,
)
from emr_hyperneat.engine.settling import _is_stable_node_fn, settle_hidden, settle_hidden_additive


class MultiTaskMixin:
    """EMRHyperNEAT's multi-task and neuromodulated evaluation."""

    # Multi-Task Evolution Support

    def _create_eval_population_true_neuromodulated(
        self,
        has_hh: bool = False,
        activate_time: int = 0,
        hidden_act_fn: Optional[Callable] = None,
        return_hidden: bool = False,
        branch_mode: str = 'none',
        positions: Optional[jnp.ndarray] = None,
    ):
        """Create vmapped true neuromodulation evaluation function.

        A vmapped function, run eagerly, that evaluates an entire population on a single task
        with a specific NT vector. Used by run_generation_multitask() for multi-task evaluation.
        It is made once per setting it closes over (H->H settling and its steps, the activation,
        the node rule, the modulation strength, the output inversion) and kept, and its node is
        a node rule whose settling step is made once (:func:`true_neuromodulation_node_rule`),
        so its settling does not compile again for every generation and task.

        Args:
            has_hh: Whether H->H connections are available
            activate_time: Number of H->H iterations (if has_hh=True)
            hidden_act_fn: Optional activation function for hidden layer.
                If None, defaults to jnp.tanh. Can be any callable from
                ACTIVATION_FUNCTIONS (e.g., jnp.sin, jax.nn.relu, etc.)
            return_hidden: Also return each genome's hidden activity, (pop, cases, positions), the
                activity W2 reads (the per-genome orthogonality bonus).
            branch_mode: multitask.branch_gating_mode: each position's modulation from the receptor
                channel its region selects (compute_modulation_with_branch_gating), in the
                feed-forward and the recurrent evaluation alike (HMR's branch-gated evaluator was
                feed-forward and tanh only).
            positions: (positions, 2) grid coordinates of the evaluated positions, which a branch
                mode needs.

        Returns:
            A vmapped function with signature:
            - If has_hh=False:
              (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
               fitness_mode, modulation_mode, modulation_strength_override)
            - If has_hh=True:
              (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
               from_indices, to_indices, hh_weights, valid_mask,
               fitness_mode, modulation_mode, modulation_strength_override)
            -> fitness_array of shape (pop_size,)
        """
        # Capture neuromodulation config in closure
        mod_strength = self.neuromod_config.modulation_strength if self.neuromod_config else 5.0
        # Use provided activation or default to tanh
        act_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh

        node_rule = self.neuromod_config.node_rule if self.neuromod_config else 'paper'
        use_output_inversion = (
            self.neuromod_config.use_output_inversion if self.neuromod_config else True
        )
        if branch_mode not in BRANCH_GATING_MODES:
            raise ValueError(
                f"branch_mode must be one of {BRANCH_GATING_MODES}, got {branch_mode!r}"
            )
        if branch_mode != 'none' and positions is None:
            raise ValueError("a branch gating mode needs the evaluated positions' grid coordinates")
        settles = bool(has_hh and activate_time > 0)
        additive = settles and self._settling_rule == 'hmr'
        key = (settles, int(activate_time) if settles else 0, act_fn, node_rule, mod_strength,
               use_output_inversion, bool(return_hidden), additive)
        # Kept without branch gating (whose positions are an array) and for an activation made once
        # (the registry's): a function made per call would add an entry per call
        keep = branch_mode == 'none' and _is_stable_node_fn(act_fn)
        if keep and key in _MULTITASK_EVALUATORS:
            return _MULTITASK_EVALUATORS[key]

        def modulated_node(receptor_densities_single, base_gains_single, neurotransmitter,
                           modulation_mode, modulation_strength_override):
            """A node under neuromodulation as a node rule and its operands: the same rule as the
            single-task path (true_neuromodulation_node_rule, neuromodulation.node_rule). The
            feed-forward and the recurrent evaluation below both apply it."""
            strength = (modulation_strength_override if modulation_strength_override is not None
                        else mod_strength)
            modulation = None if branch_mode == 'none' else compute_modulation_with_branch_gating(
                receptor_densities_single, neurotransmitter, positions, branch_mode)
            return true_neuromodulation_node_rule(
                (act_fn, ()),
                receptor_densities_single,
                base_gains_single,
                neurotransmitter,
                strength,
                neurotransmitter.shape[-1],
                modulation_mode,
                node_rule,
                modulation,
            )

        def finish(hidden, W2_single, neurotransmitter, targets, fitness_mode):
            # Output inversion by NT[3] (complementary tasks like NAND, NOR), only with four or more
            # channels and use_output_inversion (with fewer, JAX would clamp nt[3] to the last
            # channel)
            outputs = invert_outputs(jax.nn.sigmoid(hidden @ W2_single), neurotransmitter,
                                     use_output_inversion, neurotransmitter.shape[-1], node_rule)
            return compute_fitness(outputs, targets, fitness_mode)

        if settles:
            def eval_single_network_true_neuromodulated_hh(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets,
                from_idx, to_idx, hh_w, hh_mask,
                fitness_mode='mse', modulation_mode='full', modulation_strength_override=None
            ):
                """One network with TRUE neuromodulation and H->H connections, on one task.

                W1, W2, receptors, base gains and the H->H connections are fixed; only the
                neurotransmitter vector varies per task. The hidden layer settles by the
                recurrent path's one rule (:func:`settle_hidden`), with the node function of
                the feed-forward evaluation, so without H->H connections both agree.
                """
                node_fn, node_operands = modulated_node(receptor_densities_single,
                                                        base_gains_single, neurotransmitter,
                                                        modulation_mode,
                                                        modulation_strength_override)
                if additive:
                    # HMR FULL's multi-task rule (recurrence.settling_rule 'hmr'): the node of the
                    # drive, then activate_time steps of h + node(recurrent(h))
                    hidden = settle_hidden_additive(
                        node_fn(inputs @ W1_single, *node_operands),
                        from_idx,
                        to_idx,
                        hh_w,
                        hh_mask,
                        activate_time,
                        node_fn,
                        node_operands,
                    )
                else:
                    hidden = settle_hidden(inputs @ W1_single, from_idx, to_idx, hh_w, hh_mask,
                                           activate_time, node_fn, node_operands=node_operands)
                fitness = finish(hidden, W2_single, neurotransmitter, targets, fitness_mode)
                return (fitness, hidden) if return_hidden else fitness

            # Vmap over population (axis 0 for all per-genome arrays)
            eval_population = jax.vmap(
                eval_single_network_true_neuromodulated_hh,
                in_axes=(0, 0, 0, 0, None, None, None, 0, 0, 0, 0, None, None, None)
            )

        else:
            def eval_single_network_true_neuromodulated(
                W1_single, W2_single, receptor_densities_single, base_gains_single,
                neurotransmitter, inputs, targets, fitness_mode='mse',
                modulation_mode='full', modulation_strength_override=None
            ):
                """One network with TRUE neuromodulation (feedforward), on one task.

                W1, W2, receptors and base gains are fixed; only the neurotransmitter vector
                varies per task, so one network produces different behaviors.
                """
                node_fn, node_operands = modulated_node(receptor_densities_single,
                                                        base_gains_single, neurotransmitter,
                                                        modulation_mode,
                                                        modulation_strength_override)
                hidden = node_fn(inputs @ W1_single, *node_operands)
                fitness = finish(hidden, W2_single, neurotransmitter, targets, fitness_mode)
                return (fitness, hidden) if return_hidden else fitness

            # Vmap over population dimension (axis 0 for W1, W2, receptor_densities, base_gains)
            eval_population = jax.vmap(
                eval_single_network_true_neuromodulated,
                in_axes=(0, 0, 0, 0, None, None, None, None, None, None)
            )

        if keep:
            _MULTITASK_EVALUATORS[key] = eval_population
        return eval_population

    def _create_eval_population_two_module(
        self,
        all_positions: jnp.ndarray,
        hidden_act_fn: Optional[Callable] = None,
    ):
        """Create vmapped two-module evaluation function (PFC/Sensorimotor separation).

        This creates a JIT-compiled vmapped function that evaluates networks using
        the two-module architecture from Liu & Wang (2024):
        - PFC module (x < 0): Rule maintenance, uses NT[0:2]
        - Sensorimotor module (x >= 0): Task execution, uses NT[1:3]

        Args:
            all_positions: Spatial coordinates of all positions, shape (total_positions, 2)
            hidden_act_fn: Optional activation function for hidden layer.

        Returns:
            A vmapped function with signature:
            (W1, W2, receptor_densities, base_gains, NT_vector, inputs, targets,
             fitness_mode, modulation_mode, modulation_strength_override, two_module_mode)
            -> fitness_array of shape (pop_size,)
        """
        # Capture neuromodulation config in closure
        mod_strength = self.neuromod_config.modulation_strength if self.neuromod_config else 5.0
        # Use provided activation or default to tanh
        act_fn = hidden_act_fn if hidden_act_fn is not None else jnp.tanh

        # Precompute spatial masks (these are constant for all evaluations)
        x_coords = all_positions[:, 0]
        pfc_mask = x_coords < 0.0  # Left hemisphere = PFC (rule)
        sm_mask = x_coords >= 0.0   # Right hemisphere = Sensorimotor (execution)

        use_output_inversion = (self.neuromod_config.use_output_inversion if self.neuromod_config
                                else True)

        def eval_single_network_two_module(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            two_module_mode='parallel'
        ):
            """Evaluate network with two-module architecture.

            Partitions hidden layer into PFC and Sensorimotor modules based on
            spatial x-coordinate. Each module uses different NT components.

            Architecture:
                Inputs -> PFC (x<0) -+-> Outputs
                      |             |
                      `-> SM (x>=0) -+
                           ^
                           | (sequential mode: PFC feeds into SM)

            Args:
                W1_single: Input->Hidden weights, shape (num_inputs, total_positions)
                W2_single: Hidden->Output weights, shape (total_positions, num_outputs)
                receptor_densities_single: Per-neuron NT sensitivity, shape (total_positions, num_nt)
                base_gains_single: Base activation gain per neuron, shape (total_positions,)
                neurotransmitter: NT levels, shape (num_nt,)
                inputs: Input data, shape (num_cases, num_inputs)
                targets: Target data, shape (num_cases, num_outputs)
                fitness_mode: Fitness function ('mse', 'accuracy', etc.)
                modulation_mode: 'full', 'gating_only', 'gain_bias_only'
                modulation_strength_override: Override default modulation strength
                two_module_mode: 'parallel' or 'sequential'

            Returns:
                Fitness score based on fitness_mode
            """
            strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

            # PFC module (rule maintenance)
            # Uses NT[0:2] for modulation - distinct rule encoding
            W1_pfc = W1_single[:, pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]

            pre_hidden_pfc = inputs @ W1_pfc  # (num_cases, n_pfc)
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]  # NT[0:2] for PFC

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * strength)
                hidden_pfc = act_fn(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:  # 'full' mode
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # Sensorimotor module (task execution)
            # Uses NT[1:3] for modulation - overlapping NT component enables coordination
            W1_sm = W1_single[:, sm_mask]
            W2_sm = W2_single[sm_mask, :]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]

            pre_hidden_sm = inputs @ W1_sm  # (num_cases, n_sm)

            # Sequential mode: Add top-down PFC signal to sensorimotor input
            if two_module_mode == 'sequential':
                # Mean PFC activation provides top-down rule signal
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)  # (num_cases, 1)
                pre_hidden_sm = pre_hidden_sm + pfc_mean  # Broadcast to all SM positions

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]  # NT[1:3] for SM

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * strength)
                hidden_sm = act_fn(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:  # 'full' mode
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # Combine outputs
            output_pfc = hidden_pfc @ W2_pfc  # (num_cases, num_outputs)
            output_sm = hidden_sm @ W2_sm    # (num_cases, num_outputs)
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            # Output inversion by NT[3], only with four or more channels and use_output_inversion
            outputs = invert_outputs(outputs, neurotransmitter, use_output_inversion,
                                     neurotransmitter.shape[-1])

            return compute_fitness(outputs, targets, fitness_mode)

        def eval_single_network_two_module_with_hidden(
            W1_single, W2_single, receptor_densities_single, base_gains_single,
            neurotransmitter, inputs, targets,
            fitness_mode='mse', modulation_mode='full', modulation_strength_override=None,
            two_module_mode='parallel'
        ):
            """Two-module evaluation that also returns hidden activations."""
            strength = modulation_strength_override if modulation_strength_override is not None else mod_strength

            # PFC module
            W1_pfc = W1_single[:, pfc_mask]
            rd_pfc = receptor_densities_single[pfc_mask, :]
            bg_pfc = base_gains_single[pfc_mask]
            W2_pfc = W2_single[pfc_mask, :]

            pre_hidden_pfc = inputs @ W1_pfc
            mod_pfc = rd_pfc[:, :2] @ neurotransmitter[:2]

            if modulation_mode == 'gating_only':
                gates_pfc = jax.nn.sigmoid(mod_pfc * strength)
                hidden_pfc = act_fn(pre_hidden_pfc) * gates_pfc
            elif modulation_mode == 'gain_bias_only':
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
            else:
                effective_gains_pfc = bg_pfc + strength * mod_pfc
                effective_gains_pfc = jnp.clip(effective_gains_pfc, 0.1, 5.0)
                effective_gains_pfc = jnp.where(jnp.isnan(effective_gains_pfc), 1.0, effective_gains_pfc)
                gates_pfc = jax.nn.sigmoid(mod_pfc)
                modulation_bias_pfc = mod_pfc * strength
                hidden_pfc = act_fn(effective_gains_pfc * pre_hidden_pfc + modulation_bias_pfc)
                hidden_pfc = hidden_pfc * gates_pfc

            # Sensorimotor module
            W1_sm = W1_single[:, sm_mask]
            rd_sm = receptor_densities_single[sm_mask, :]
            bg_sm = base_gains_single[sm_mask]
            W2_sm = W2_single[sm_mask, :]

            pre_hidden_sm = inputs @ W1_sm
            if two_module_mode == 'sequential':
                pfc_mean = jnp.mean(hidden_pfc, axis=-1, keepdims=True)
                pre_hidden_sm = pre_hidden_sm + pfc_mean

            mod_sm = rd_sm[:, 1:3] @ neurotransmitter[1:3]

            if modulation_mode == 'gating_only':
                gates_sm = jax.nn.sigmoid(mod_sm * strength)
                hidden_sm = act_fn(pre_hidden_sm) * gates_sm
            elif modulation_mode == 'gain_bias_only':
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
            else:
                effective_gains_sm = bg_sm + strength * mod_sm
                effective_gains_sm = jnp.clip(effective_gains_sm, 0.1, 5.0)
                effective_gains_sm = jnp.where(jnp.isnan(effective_gains_sm), 1.0, effective_gains_sm)
                gates_sm = jax.nn.sigmoid(mod_sm)
                modulation_bias_sm = mod_sm * strength
                hidden_sm = act_fn(effective_gains_sm * pre_hidden_sm + modulation_bias_sm)
                hidden_sm = hidden_sm * gates_sm

            # Combine outputs
            output_pfc = hidden_pfc @ W2_pfc
            output_sm = hidden_sm @ W2_sm
            outputs = jax.nn.sigmoid(output_pfc + output_sm)

            # Output inversion by NT[3], only with four or more channels and use_output_inversion
            outputs = invert_outputs(outputs, neurotransmitter, use_output_inversion,
                                     neurotransmitter.shape[-1])

            fitness = compute_fitness(outputs, targets, fitness_mode)

            # Combine hidden for orthogonality computation - pad to total_positions
            total_positions = all_positions.shape[0]
            num_cases = inputs.shape[0]
            combined_hidden = jnp.zeros((num_cases, total_positions))
            combined_hidden = combined_hidden.at[:, pfc_mask].set(hidden_pfc)
            combined_hidden = combined_hidden.at[:, sm_mask].set(hidden_sm)

            return fitness, combined_hidden

        # Create vmapped versions
        eval_population_two_module = jax.vmap(
            eval_single_network_two_module,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
        )

        eval_population_two_module_with_hidden = jax.vmap(
            eval_single_network_two_module_with_hidden,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None)
        )

        return eval_population_two_module, eval_population_two_module_with_hidden

    def run_generation_multitask(
        self,
        state: Any,
        problems: List[Any],
        neurotransmitters: Optional[List[jnp.ndarray]] = None,
        aggregation_method: str = 'mean',
        aggregation_weights: Optional[List[float]] = None,
        orthogonality_bonus: float = 0.0,
        specialization_bonus: float = 0.0,
    ) -> Tuple[Any, 'AlgorithmMetrics']:
        """Run one generation of multi-task evolution with TRUE neuromodulation.

        This is the method for multi-task evolution, where:
        1. CPPN evolution happens once per generation
        2. W1, W2, receptor_densities, base_gains are built once (via run_generation_verbose)
        3. Fitness is computed on all tasks with task-specific NT vectors
        4. Aggregated multi-task fitness is used for NEAT selection
        5. Same network architecture solves multiple tasks

        The approach mirrors the implementation in HMR's neuromodulation variant (release 0.1.0).

        Args:
            state: Current algorithm state
            problems: List of problems (tasks) to evaluate on
            neurotransmitters: Optional list of NT vectors for each task.
                If None, uses default XOR, AND, OR, NAND, NOR presets.
            aggregation_method: How to combine per-task fitnesses:
                - 'mean': Average fitness across tasks
                - 'min': Minimum fitness (hardest task matters most)
                - 'weighted': Weighted average using aggregation_weights
                - 'product': Geometric mean (product^(1/n))
                - 'softmin': Soft minimum (smoother than min)
                - 'harmonic': Harmonic mean (emphasizes poor performance)
            aggregation_weights: Weights for 'weighted' method (must sum to 1)
            orthogonality_bonus: Refused above 0, as specialization_bonus (see Raises).
            specialization_bonus: Refused above 0.

        Returns:
            Tuple of (new_state, metrics) with multi-task metrics

        Raises:
            ValueError: If neuromodulation is disabled or mode not true_neuromodulation; a bonus
                above 0 (the two arguments): each adds one value to every genome of a
                generation, from the NT vectors or from the population's mean variance, so it
                never changes a rank within the generation, only the reported fitness (the
                configuration keys are refused as well; the per-genome bonuses are
                orthogonality_bonus_weight and specialization_bonus_weight); a branch gating mode
                or modulation penalty that cannot apply (check_branch_gating_and_penalty),
                refused before the generation runs; the node-list model (network_model
                'node_list'), which evaluates one task.
        """
        import time as time_module

        if self.network_model == 'node_list':
            raise ValueError("run_generation_multitask evaluates the grid model's networks; "
                             "recurrence.network_model 'node_list' evaluates one task "
                             "(run_generation)")

        if isinstance(problems, MultiTaskConfig):
            # HMR's call form: run_generation_multitask(state, MultiTaskConfig(tasks={name: (inputs,
            # targets)}, ...))
            return self._run_generation_multitask_tasks(state, problems)

        total_start = time_module.perf_counter()

        self._refuse_palette_evolution('run_generation_multitask')
        for name, value in (('orthogonality_bonus', orthogonality_bonus),
                            ('specialization_bonus', specialization_bonus)):
            if value:
                raise ValueError(f"{name}={value} would add one value to every genome of a generation, which "
                                 f"never changes a rank within it; the per-genome bonuses are "
                                 f"multitask.orthogonality_bonus_weight and "
                                 f"specialization_bonus_weight")
        # The same settings on a configuration object set by hand or by HMR's call form, where
        # create_config's checks never ran (0.1.0 ignored them), and the per-node features 0.1.0
        # dropped on a direct call without the multitask block
        _set = getattr(self, 'multitask_config', None)
        if _set is not None:
            for name in ('orthogonality_bonus', 'specialization_bonus'):
                value = getattr(_set, name, 0.0)
                if value:
                    raise ValueError(
                        f"MultiTaskConfig.{name}={value} would add one value to every genome of a "
                        "generation, which never changes a rank within it; the per-genome bonuses "
                        "are orthogonality_bonus_weight and specialization_bonus_weight"
                    )
            if getattr(_set, 'confusion_eval_frequency', 0):
                raise ValueError(
                    f"MultiTaskConfig.confusion_eval_frequency="
                    f"{_set.confusion_eval_frequency!r} is never read (HMR's default was 1 and HMR "
                    f"never read it either); pass 0"
                )
            if getattr(_set, 'generalist_bonus_type', 'none') not in GENERALIST_BONUS_TYPES:
                raise ValueError(
                    f"MultiTaskConfig.generalist_bonus_type must be one of "
                    f"{GENERALIST_BONUS_TYPES}, got {_set.generalist_bonus_type!r} (no bonus "
                    f"applied)"
                )
        _refuse_per_node_in_multitask(
            [
                feature
                for feature in self._features_a_tanh_sum_path_drops()
                if feature not in ("neuromodulation", "recurrence.settling_rule='hmr'")
            ]
        )

        # Validate neuromodulation is enabled
        if self.neuromod_config is None or self.neuromod_config.mode == 'disabled':
            raise ValueError(
                "run_generation_multitask() requires neuromodulation to be enabled. "
                "Set 'neuromodulation': {'mode': 'true_neuromodulation', ...} in config."
            )

        if self.neuromod_config.mode != 'true_neuromodulation':
            raise ValueError(
                f"run_generation_multitask() requires 'true_neuromodulation' mode, "
                f"got '{self.neuromod_config.mode}'. Use run_generation_verbose() "
                f"for other neuromodulation modes."
            )

        num_tasks = len(problems)
        # Branch gating and the modulation penalty (HMR's), checked before the generation runs
        _configured = getattr(self, 'multitask_config', None)
        cfg_branch_gating_mode = (
            getattr(_configured, 'branch_gating_mode', 'none') if _configured else 'none'
        )
        cfg_modulation_penalty = (
            getattr(_configured, 'modulation_penalty', 0.0) if _configured else 0.0
        )
        check_branch_gating_and_penalty(
            cfg_branch_gating_mode, cfg_modulation_penalty,
            getattr(_configured, 'two_module_mode', 'none') if _configured else 'none',
            self.neuromod_config.num_nt_types, num_tasks)
        refuse_anti_branch_gating(self.neuromod_config.node_rule, cfg_branch_gating_mode)

        # Setup NT vectors
        if neurotransmitters is None:
            # Use default task presets
            task_names = ['xor', 'and', 'or', 'nand', 'nor']
            neurotransmitters = [
                NT_TASK_PRESETS.get(task_names[i % len(task_names)], NT_TASK_PRESETS['xor'])
                for i in range(num_tasks)
            ]
        elif len(neurotransmitters) != num_tasks:
            raise ValueError(
                f"Number of NT vectors ({len(neurotransmitters)}) must match "
                f"number of problems ({num_tasks})"
            )

        # Validate aggregation weights
        if aggregation_method == 'weighted':
            if aggregation_weights is None:
                aggregation_weights = [1.0 / num_tasks] * num_tasks
            elif len(aggregation_weights) != num_tasks:
                raise ValueError(
                    f"Number of weights ({len(aggregation_weights)}) must match "
                    f"number of problems ({num_tasks})"
                )
            # Normalize weights
            total_weight = sum(aggregation_weights)
            aggregation_weights = [w / total_weight for w in aggregation_weights]

        # A mode set on the configuration object by hand is checked before the generation is built
        mode = getattr(self.multitask_config, 'fitness_mode', 'mse') if self.multitask_config else 'mse'
        if mode not in MULTITASK_FITNESS_MODES:
            raise ValueError(f"multitask fitness_mode must be one of {MULTITASK_FITNESS_MODES}, got {mode!r}")

        # Store current multi-task state
        self._multitask_state['num_tasks'] = num_tasks
        self._multitask_state['task_names'] = [
            getattr(p, 'name', f'task_{i}') for i, p in enumerate(problems)
        ]

        # Call run_generation_verbose once to build W1, W2 and cache them
        # Use first problem to build weight matrices
        # This caches: self._cached_W1, self._cached_W2, self._neuromod_true
        first_problem = problems[0]
        # Its metrics carry this population's topology, which no task changes.
        _, structure = self.run_generation_verbose(state, first_problem, skip_metrics=True)
        # That generation's champion is the first task's, not this generation's
        self._last_eval = None

        # Extract cached data
        if not hasattr(self, '_cached_W1') or self._cached_W1 is None:
            raise RuntimeError(
                "Weight matrices not cached. Ensure run_generation_verbose() was called "
                "with TRUE neuromodulation enabled."
            )
        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError(
                "TRUE neuromodulation data not available. Check configuration."
            )

        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']

        # Check for H->H connections and set up activation functions
        # Check for cached sparse H->H connections from run_generation_verbose
        sparse_hh = getattr(self, '_cached_sparse_hh', None)
        has_hh = (
            sparse_hh is not None and
            self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH
        )
        activate_time = self.activate_time if has_hh else 0

        # Determine per-task activation functions from config
        # Priority: per_task_activation > hidden_activation > 'tanh' (default)
        multitask_cfg = getattr(self, 'multitask_config', None)
        per_task_act_names = []  # List of activation names per task
        task_names = self._multitask_state.get('task_names', [f'task_{i}' for i in range(num_tasks)])

        for task_name in task_names:
            if multitask_cfg is not None and multitask_cfg.per_task_activation:
                # Per-task activation specified
                act_name = multitask_cfg.per_task_activation.get(task_name, 'tanh')
            elif multitask_cfg is not None and multitask_cfg.hidden_activation:
                # Global activation specified
                act_name = multitask_cfg.hidden_activation
            else:
                # Default to tanh for backwards compatibility
                act_name = 'tanh'
            per_task_act_names.append(act_name)

        # Extract extended config values (ported from neuromodulation_functions.py)
        # These use safe defaults to preserve existing behavior
        cfg_fitness_mode = 'mse'
        cfg_modulation_mode = 'full'
        cfg_modulation_strength_override = None
        cfg_two_module_mode = 'none'
        cfg_orthogonality_bonus_weight = orthogonality_bonus  # Use function arg as default
        cfg_orthogonality_metric = 'cosine_mean'
        cfg_generalist_bonus_type = 'none'
        cfg_generalist_bonus_weight = 0.0
        cfg_generalist_threshold = 0.9
        cfg_softmin_temperature = 0.1
        cfg_specialization_bonus_weight = 0.0

        if multitask_cfg is not None:
            cfg_fitness_mode = getattr(multitask_cfg, 'fitness_mode', 'mse')
            cfg_modulation_mode = getattr(multitask_cfg, 'modulation_mode', 'full')
            cfg_modulation_strength_override = getattr(multitask_cfg, 'modulation_strength_override', None)
            cfg_two_module_mode = getattr(multitask_cfg, 'two_module_mode', 'none')
            # Use config orthogonality_bonus_weight if set, otherwise use function arg
            if hasattr(multitask_cfg, 'orthogonality_bonus_weight') and multitask_cfg.orthogonality_bonus_weight > 0:
                cfg_orthogonality_bonus_weight = multitask_cfg.orthogonality_bonus_weight
            cfg_orthogonality_metric = getattr(multitask_cfg, 'orthogonality_metric', 'cosine_mean')
            cfg_generalist_bonus_type = getattr(multitask_cfg, 'generalist_bonus_type', 'none')
            cfg_generalist_bonus_weight = getattr(multitask_cfg, 'generalist_bonus_weight', 0.0)
            cfg_generalist_threshold = getattr(multitask_cfg, 'generalist_threshold', 0.9)
            cfg_softmin_temperature = getattr(multitask_cfg, 'softmin_temperature', 0.1)
            cfg_specialization_bonus_weight = (
                getattr(multitask_cfg, 'specialization_bonus_weight', 0.0) or 0.0
            )
        if cfg_orthogonality_metric not in ORTHOGONALITY_METRICS:
            raise ValueError(
                f"multitask.orthogonality_metric must be one of {ORTHOGONALITY_METRICS}, got "
                f"{cfg_orthogonality_metric!r}"
            )
        if cfg_specialization_bonus_weight > 0.0 and len(problems) < 2:
            raise ValueError(
                "the specialization bonus compares a task's NT vector with another task's: it "
                "needs two or more tasks"
            )

        # Also checked by create_config; repeated here because a runner may set two_module_mode on
        # the MultiTaskConfig after create_config
        if cfg_two_module_mode != 'none' and has_hh:
            raise ValueError(
                "multitask.two_module_mode evaluates feed-forward only and would ignore the "
                "hidden-to-hidden connections this configuration discovers"
            )
        if (cfg_two_module_mode != 'none' and self.neuromod_config is not None
                and self.neuromod_config.node_rule != 'paper'):
            raise ValueError(
                f"neuromodulation.node_rule {self.neuromod_config.node_rule!r} is not applied by "
                f"multitask.two_module_mode, whose modules compute the paper's rule"
            )

        # The evaluator of each activation the tasks use (made once per setting and kept, see
        # _create_eval_population_true_neuromodulated)
        unique_activations = set(per_task_act_names)
        eval_fn_cache = {}
        # With the hidden activity, for the per-genome orthogonality bonus
        eval_fn_cache_hidden = {}
        eval_fn_cache_two_module = {}  # For two-module mode

        # Get all_positions for two-module evaluation (if needed)
        all_positions = None
        if cfg_two_module_mode != 'none':
            # The grid itself
            h_grid = get_hierarchical_grid(self.max_depth)
            if h_grid is not None:
                all_positions = h_grid.all_positions
            else:
                # Fall back to standard mode if positions not available
                cfg_two_module_mode = 'none'
                if self.verbose:
                    print("[MultiTask] Warning: two_module_mode requires cached positions, falling "
                          "back to 'none'")

        # The grid coordinates of the evaluated positions, for branch gating: the multi-task
        # evaluation runs on the whole grid (compaction is blocked while a multi-task configuration
        # is on)
        branch_positions = None
        if cfg_branch_gating_mode != 'none':
            branch_positions = get_hierarchical_grid(self.max_depth).all_positions
            if branch_positions.shape[0] != W1.shape[-1]:
                raise RuntimeError(
                    "branch gating needs the evaluation on the whole grid "
                    f"({branch_positions.shape[0]} "
                    f"positions); the cached weights have {W1.shape[-1]}"
                )

        for act_name in unique_activations:
            # An unknown name (a MultiTaskConfig built by hand; create_config checks the keys)
            # raises rather than running tanh
            if act_name not in ACTIVATION_FUNCTIONS:
                raise ValueError(f"unknown multi-task activation {act_name!r}; the names are "
                                 f"{sorted(ACTIVATION_FUNCTIONS)}")
            act_fn = ACTIVATION_FUNCTIONS[act_name]
            eval_fn_cache[act_name] = self._create_eval_population_true_neuromodulated(
                has_hh=has_hh,
                activate_time=activate_time,
                hidden_act_fn=act_fn,
                branch_mode=cfg_branch_gating_mode,
                positions=branch_positions,
            )
            if cfg_orthogonality_bonus_weight > 0.0:
                eval_fn_cache_hidden[act_name] = self._create_eval_population_true_neuromodulated(
                    has_hh=has_hh,
                    activate_time=activate_time,
                    hidden_act_fn=act_fn,
                    return_hidden=True,
                    branch_mode=cfg_branch_gating_mode,
                    positions=branch_positions,
                )
            # Also create two-module eval function if needed
            if cfg_two_module_mode != 'none' and all_positions is not None:
                eval_fn_cache_two_module[act_name] = self._create_eval_population_two_module(
                    all_positions=all_positions,
                    hidden_act_fn=act_fn,
                )

        if self.verbose and len(unique_activations) > 1:
            print(f"[MultiTask] Using {len(unique_activations)} unique activation functions: {unique_activations}")
        if self.verbose and cfg_two_module_mode != 'none':
            print(f"[MultiTask] Using two-module architecture mode: {cfg_two_module_mode}")

        def evaluate_task(task_act_name, nt_vector, inputs, targets, with_hidden=False):
            """The population's fitness on one task's data under an NT vector, with the task's
            evaluator (two-module, recurrent or feed-forward), and its hidden activity with
            ``with_hidden``."""
            if cfg_two_module_mode != 'none' and task_act_name in eval_fn_cache_two_module:
                evaluator = eval_fn_cache_two_module[task_act_name][1 if with_hidden else 0]
                return evaluator(
                    W1,
                    W2,
                    rd,
                    bg,
                    nt_vector,
                    inputs,
                    targets,
                    cfg_fitness_mode,
                    cfg_modulation_mode,
                    cfg_modulation_strength_override,
                    cfg_two_module_mode,
                )
            evaluator = (eval_fn_cache_hidden if with_hidden else eval_fn_cache)[task_act_name]
            if has_hh and sparse_hh is not None:
                return evaluator(
                    W1,
                    W2,
                    rd,
                    bg,
                    nt_vector,
                    inputs,
                    targets,
                    sparse_hh.from_indices,
                    sparse_hh.to_indices,
                    sparse_hh.weights,
                    sparse_hh.valid_mask,
                    cfg_fitness_mode,
                    cfg_modulation_mode,
                    cfg_modulation_strength_override,
                )
            return evaluator(
                W1,
                W2,
                rd,
                bg,
                nt_vector,
                inputs,
                targets,
                cfg_fitness_mode,
                cfg_modulation_mode,
                cfg_modulation_strength_override,
            )

        # STEP 4: Evaluate on each task with different NT vectors
        per_task_fitnesses = []
        per_task_times = []
        # task index -> (pop, cases, positions) at each genome's active positions
        per_task_hidden = {}
        per_task_data = []  # (activation, inputs, targets) per task, for the specialization bonus
        if cfg_orthogonality_bonus_weight > 0.0:
            active = jnp.asarray(self._cached_masks_A, dtype=jnp.float32)[:, None, :]

        for task_idx, (problem, nt_vector) in enumerate(zip(problems, neurotransmitters)):
            task_start = time_module.perf_counter()

            # Get inputs and targets for this task
            if hasattr(problem, 'get_inputs'):
                inputs = jnp.array(problem.get_inputs())
                targets = jnp.array(problem.get_targets())
            elif hasattr(problem, 'inputs'):
                inputs = jnp.array(problem.inputs)
                targets = jnp.array(problem.targets)
            else:
                # Try get_data() method (returns list of (input, target) tuples)
                data = problem.get_data()
                inputs = jnp.array([d[0] for d in data])
                targets = jnp.array([d[1] for d in data])

            # Ensure NT vector is a JAX array
            nt_vector = jnp.array(nt_vector)

            # Get the appropriate eval function for this task's activation
            task_act_name = per_task_act_names[task_idx]
            if cfg_orthogonality_bonus_weight > 0.0:
                fitness, hidden = evaluate_task(
                    task_act_name, nt_vector, inputs, targets, with_hidden=True
                )
                # A genome's fingerprint counts its own active positions only: elsewhere W1 and W2
                # are zero, but the node's NT-dependent bias makes the activity nonzero (HMR's
                # counted it, rewarding receptors on positions that reach no output)
                per_task_hidden[task_idx] = hidden * active
            else:
                fitness = evaluate_task(task_act_name, nt_vector, inputs, targets)
            per_task_data.append((task_act_name, inputs, targets))
            per_task_fitnesses.append(fitness)
            per_task_times.append(time_module.perf_counter() - task_start)

        # Aggregate fitnesses
        stacked_fitnesses = jnp.stack(per_task_fitnesses, axis=0)  # (num_tasks, pop_size)

        if aggregation_method == 'mean':
            aggregated_fitness = jnp.mean(stacked_fitnesses, axis=0)
        elif aggregation_method == 'min':
            aggregated_fitness = jnp.min(stacked_fitnesses, axis=0)
        elif aggregation_method == 'weighted':
            weights = jnp.array(aggregation_weights)
            aggregated_fitness = jnp.sum(stacked_fitnesses * weights[:, None], axis=0)
        elif aggregation_method == 'product':
            # Geometric mean: (f1 * f2 * ... * fn)^(1/n)
            eps = 1e-8
            log_fitness = jnp.sum(jnp.log(stacked_fitnesses + eps), axis=0)
            aggregated_fitness = jnp.exp(log_fitness / num_tasks)
        elif aggregation_method == 'raw_product':
            # HMR's 'product' (its MultiTaskConfig call form): the plain product of the task
            # fitnesses. It ranks genomes like the geometric mean except that genomes with a zero
            # task score tie, and a bonus weighs differently.
            aggregated_fitness = jnp.prod(stacked_fitnesses, axis=0)
        elif aggregation_method == 'softmin':
            # Softmin: weighted average where smaller values get more weight
            # Use config temperature (lower = sharper, more like true min)
            neg_scaled = -stacked_fitnesses / cfg_softmin_temperature
            weights = jax.nn.softmax(neg_scaled, axis=0)
            aggregated_fitness = jnp.sum(stacked_fitnesses * weights, axis=0)
        elif aggregation_method == 'harmonic':
            # Harmonic mean: n / (1/f1 + 1/f2 + ... + 1/fn)
            eps = 1e-8
            reciprocal_sum = jnp.sum(1.0 / (stacked_fitnesses + eps), axis=0)
            aggregated_fitness = num_tasks / reciprocal_sum
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Per-genome bonuses (HMR's mechanisms). The constant bonuses (from the tasks' NT
        # vectors, from the population's mean fitness variance) give every genome the same value,
        # which selection never sees, and raise.
        effective_orthogonality_bonus = cfg_orthogonality_bonus_weight
        orthogonality = specialization = None
        if effective_orthogonality_bonus > 0.0:
            # Each genome's task fingerprints (its hidden activity averaged over a task's cases),
            # compared across tasks
            orthogonality = compute_population_orthogonality(
                {f'task_{t}': per_task_hidden[t] for t in range(num_tasks)},
                cfg_orthogonality_metric,
            )
            aggregated_fitness = aggregated_fitness + effective_orthogonality_bonus * orthogonality
        if cfg_specialization_bonus_weight > 0.0:
            # A genome's fitness under each task's own NT vector minus its fitness on the same data
            # under other tasks' vectors, with the same evaluator: HMR's used tasks (t+1)%T and
            # (t+2)%T, so with two tasks one of its "wrong" vectors was the right one, and evaluated
            # them without the two-module or branch mode the right one used. Two tasks: the other
            # task's vector; three or more: the next two.
            wrong = []
            for t, (task_act_name, inputs, targets) in enumerate(per_task_data):
                for k in (
                    [(t + 1) % num_tasks]
                    if num_tasks == 2
                    else [(t + 1) % num_tasks, (t + 2) % num_tasks]
                ):
                    wrong.append(
                        evaluate_task(
                            task_act_name, jnp.array(neurotransmitters[k]), inputs, targets
                        )
                    )
            specialization = jnp.mean(stacked_fitnesses, axis=0) - jnp.mean(
                jnp.stack(wrong, axis=0), axis=0
            )
            aggregated_fitness = (
                aggregated_fitness + cfg_specialization_bonus_weight * specialization
            )
        modulation_variance = None
        if cfg_modulation_penalty > 0.0:
            # HMR's modulation_penalty, a bonus despite its name: penalty times the variance over
            # tasks of a genome's mean modulation, so a genome whose receptors answer the tasks' NT
            # vectors differently scores higher. The mean is over the genome's own active positions,
            # of the modulation its nodes receive (routed under branch gating): HMR averaged the
            # receptors over every position of the grid, those that reach no output included, and
            # multiplied them with every channel of the NT vectors, the inversion channel nt[3]
            # included, whatever the branch mode.
            active = jnp.asarray(self._cached_masks_A, dtype=bool)
            count = jnp.maximum(jnp.sum(active, axis=1), 1)
            k = min(3, self.neuromod_config.num_nt_types)
            mean_modulation = []
            for nt_vector in neurotransmitters:
                nt = jnp.asarray(nt_vector, dtype=jnp.float32)
                if cfg_branch_gating_mode == 'none':
                    modulation = rd[:, :, :k] @ nt[:k]
                else:
                    modulation = jax.vmap(lambda receptors: compute_modulation_with_branch_gating(
                        receptors, nt, branch_positions, cfg_branch_gating_mode))(rd)
                mean_modulation.append(jnp.sum(jnp.where(active, modulation, 0.0), axis=1) / count)
            modulation_variance = jnp.var(jnp.stack(mean_modulation, axis=0), axis=0)
            aggregated_fitness = aggregated_fitness + cfg_modulation_penalty * modulation_variance

        # Apply generalist bonus (ported from neuromodulation_functions.py)
        if cfg_generalist_bonus_type != 'none' and cfg_generalist_bonus_weight > 0.0:
            if cfg_generalist_bonus_type == 'min_bonus':
                # Bonus based on minimum fitness across tasks (rewards consistent performance)
                min_fitnesses = jnp.min(stacked_fitnesses, axis=0)
                generalist_bonus = cfg_generalist_bonus_weight * min_fitnesses
                aggregated_fitness = aggregated_fitness + generalist_bonus
            elif cfg_generalist_bonus_type == 'variance_penalty':
                # Penalty for high variance across tasks (encourages balanced performance)
                fitness_variance_per_ind = jnp.var(stacked_fitnesses, axis=0)
                variance_penalty = cfg_generalist_bonus_weight * fitness_variance_per_ind
                aggregated_fitness = aggregated_fitness - variance_penalty
            elif cfg_generalist_bonus_type == 'threshold_bonus':
                # Bonus if all tasks exceed threshold
                above_threshold = stacked_fitnesses >= cfg_generalist_threshold
                all_above = jnp.all(above_threshold, axis=0)
                threshold_bonus = jnp.where(all_above, cfg_generalist_bonus_weight, 0.0)
                aggregated_fitness = aggregated_fitness + threshold_bonus

        # Run NEAT selection with aggregated fitness
        # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees it.
        aggregated_fitness, nan_fitness_count = sanitize_fitness(aggregated_fitness)
        new_state = self._compiled_tell(state, aggregated_fitness)
        # Kept for the call form's MultiTaskMetrics (per task name), with the per-genome bonuses
        self._last_multitask = {'per_task': per_task_fitnesses, 'aggregated': aggregated_fitness,
                                'orthogonality': orthogonality, 'specialization': specialization,
                                'modulation_variance': modulation_variance}

        # The generation counter was advanced by the run_generation_verbose call above; a
        # second increment here made the H->H cache refresh and the random modes' per-
        # generation keys run twice as fast under multi-task evaluation.

        # Build multi-task metrics
        total_time = time_module.perf_counter() - total_start

        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            aggregated_fitness, nan_fitness_count)

        # Find best generalist (single individual with highest aggregated fitness)
        best_gen_idx = int(jnp.argmax(aggregated_fitness))
        best_generalist_per_task = {
            f'task_{i}': float(per_task_fitnesses[i][best_gen_idx])
            for i in range(num_tasks)
        }

        custom_metrics = {
            'method': 'multitask',
            'num_tasks': num_tasks,
            'nan_fitness_count': nan_fitness_count,
            'aggregation_method': aggregation_method,
            'per_task_best': [float(jnp.max(f)) for f in per_task_fitnesses],
            'per_task_mean': [float(jnp.mean(f)) for f in per_task_fitnesses],
            'per_task_times_ms': [t * 1000 for t in per_task_times],
            'best_generalist_idx': best_gen_idx,
            'best_generalist_per_task': best_generalist_per_task,
            'orthogonality_bonus_applied': (
                effective_orthogonality_bonus if effective_orthogonality_bonus > 0 else None
            ),
            'orthogonality_mean': None if orthogonality is None else float(jnp.mean(orthogonality)),
            'specialization_bonus_applied': (
                cfg_specialization_bonus_weight if cfg_specialization_bonus_weight > 0 else None
            ),
            'specialization_gap_mean': (
                None if specialization is None else float(jnp.mean(specialization))
            ),
            'branch_gating_mode': (
                cfg_branch_gating_mode if cfg_branch_gating_mode != 'none' else None
            ),
            'modulation_penalty_applied': (
                cfg_modulation_penalty if cfg_modulation_penalty > 0 else None
            ),
            'modulation_variance_mean': (
                None if modulation_variance is None else float(jnp.mean(modulation_variance))
            ),
            # Extended features (ported from neuromodulation_functions.py)
            'fitness_mode': cfg_fitness_mode,
            'modulation_mode': cfg_modulation_mode,
            'two_module_mode': cfg_two_module_mode if cfg_two_module_mode != 'none' else None,
            'orthogonality_metric': cfg_orthogonality_metric,
            'generalist_bonus_type': (
                cfg_generalist_bonus_type if cfg_generalist_bonus_type != 'none' else None
            ),
            'generalist_bonus_weight': (
                cfg_generalist_bonus_weight if cfg_generalist_bonus_weight > 0 else None
            ),
        }

        # Update fitness history for backward transfer tracking
        if 'fitness_history' not in self._multitask_state:
            self._multitask_state['fitness_history'] = []
        self._multitask_state['fitness_history'].append(
            [float(jnp.max(f)) for f in per_task_fitnesses]
        )

        total_time = time_module.perf_counter() - total_start
        num_species, species_sizes, species_fitness = species_summary(state.species,
                                                                      aggregated_fitness)
        metrics = AlgorithmMetrics(
            generation=self._current_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=len(aggregated_fitness) * num_tasks,  # pop_size * num_tasks
            time_elapsed=total_time,
            custom_metrics=custom_metrics,
            hidden_nodes_mean=structure.hidden_nodes_mean,
            hidden_nodes_std=structure.hidden_nodes_std,
            connections_mean=structure.connections_mean,
            connections_std=structure.connections_std,
        )

        return new_state, metrics

    def _run_generation_multitask_tasks(
        self, state: Any, config: 'MultiTaskConfig'
    ) -> Tuple[Any, MultiTaskMetrics]:
        """One multi-task generation in HMR's call form, through the list form.

        HMR's runners pass a MultiTaskConfig whose ``tasks`` maps task names to (inputs, targets),
        with the NT vector of each task from ``nt_vectors`` or, by name, from the NT presets
        (get_nt_for_task, HMR's lookup), weights by name for 'weighted' (a missing name weighs 1.0,
        as in HMR), and HMR's 'product', the plain product of the task fitnesses (EMR's list form
        names that 'raw_product'; its 'product' is the geometric mean). The configuration is the
        call's: it replaces the model's multi-task configuration for the generation, which also
        keeps the evaluation on the whole grid (compaction would narrow the receptors and not the
        cached weights).

        Returns:
            ``(new_state, MultiTaskMetrics)`` keyed by task name.

        Raises:
            ValueError: no tasks, a confusion_eval_frequency other than 0 (never read), an NT
                vector of the wrong size, 'weighted' without weights, or a setting the list form
                refuses (check_branch_gating_and_penalty, the bonuses' checks, the constant
                bonuses, an unknown generalist_bonus_type).
        """
        import dataclasses
        from types import SimpleNamespace

        if not config.tasks:
            raise ValueError("MultiTaskConfig.tasks must map task names to (inputs, targets)")
        if config.orthogonality_metric != 'cosine_mean' and not config.orthogonality_bonus_weight:
            raise ValueError("MultiTaskConfig.orthogonality_metric applies only with "
                             "orthogonality_bonus_weight above 0")
        if config.confusion_eval_frequency:
            # 0 only, as in the configuration's multitask block (HMR's default was 1)
            raise ValueError(
                f"MultiTaskConfig.confusion_eval_frequency={config.confusion_eval_frequency!r} "
                f"is never read (HMR's default was 1 and HMR never read it either); pass 0"
            )
        names = list(config.tasks)
        problems = [SimpleNamespace(name=name, inputs=np.asarray(data[0], dtype=np.float32),
                                    targets=np.asarray(data[1], dtype=np.float32))
                    for name, data in config.tasks.items()]
        num_nt = self.neuromod_config.num_nt_types
        neurotransmitters = []
        for name in names:
            given = (config.nt_vectors or {}).get(name)
            nt = (
                get_nt_for_task(name, num_nt)
                if given is None
                else jnp.asarray(given, dtype=jnp.float32)
            )
            if nt.shape != (num_nt,):
                raise ValueError(
                    f"the NT vector of task {name!r} has shape {nt.shape}; neuromodulation has "
                    f"{num_nt} types"
                )
            neurotransmitters.append(nt)
        aggregation = config.fitness_aggregation
        weights = None
        if aggregation == 'weighted':
            if config.weights is None:
                raise ValueError("aggregation 'weighted' needs weights by task name")
            weights = [float(config.weights.get(name, 1.0)) for name in names]
        elif config.weights is not None:
            raise ValueError(
                "MultiTaskConfig.weights is read by aggregation 'weighted' only, not "
                f"{aggregation!r}"
            )
        if aggregation == 'product':
            aggregation = 'raw_product'
        previous = self.multitask_config
        self.multitask_config = dataclasses.replace(
            config, enabled=True, num_tasks=len(names), task_names=names
        )
        try:
            new_state, _ = self.run_generation_multitask(
                state, problems, neurotransmitters, aggregation, weights
            )
        finally:
            self.multitask_config = previous
        per_task = [np.asarray(f, dtype=np.float64) for f in self._last_multitask['per_task']]
        aggregated = np.asarray(self._last_multitask['aggregated'], dtype=np.float64)
        finite = aggregated[np.isfinite(aggregated)]
        best = int(np.argmax(aggregated))
        metrics = MultiTaskMetrics(
            aggregated_fitness=float(finite.max()) if finite.size else float('nan'),
            mean_aggregated_fitness=float(finite.mean()) if finite.size else float('nan'),
            per_task_best={name: float(np.nanmax(f)) for name, f in zip(names, per_task)},
            per_task_mean={name: float(np.nanmean(f)) for name, f in zip(names, per_task)},
            best_generalist_per_task={name: float(f[best]) for name, f in zip(names, per_task)},
            best_generalist_idx=best,
            generation=self._current_generation,
        )
        return new_state, metrics

    def _eval_population_true_neuromodulated(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        receptor_densities: jnp.ndarray,
        base_gains: jnp.ndarray,
        neurotransmitter: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
        modulation_strength_override: Optional[float] = None,
    ) -> jnp.ndarray:
        """HMR's name for the feed-forward population evaluator under true neuromodulation (its
        runners call it with the cached weights and receptors after a generation): the evaluator
        run_generation_multitask uses, tanh hidden nodes. The arguments are HMR's, in its order.

        Returns:
            (pop_size,) fitness.
        """
        if getattr(self, '_true_neuromodulation_evaluator', None) is None:
            self._true_neuromodulation_evaluator = (
                self._create_eval_population_true_neuromodulated()
            )
        return self._true_neuromodulation_evaluator(
            W1, W2, receptor_densities, base_gains, neurotransmitter, inputs, targets,
            fitness_mode, modulation_mode, modulation_strength_override)

    def _eval_population_true_neuromodulated_with_hidden(
        self,
        W1: jnp.ndarray,
        W2: jnp.ndarray,
        receptor_densities: jnp.ndarray,
        base_gains: jnp.ndarray,
        neurotransmitter: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
        modulation_strength_override: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """HMR's name: _eval_population_true_neuromodulated that also returns each genome's hidden
        activity.

        Returns:
            (pop_size,) fitness and (pop_size, cases, positions) hidden activity.
        """
        if getattr(self, '_true_neuromodulation_evaluator_hidden', None) is None:
            self._true_neuromodulation_evaluator_hidden = (
                self._create_eval_population_true_neuromodulated(return_hidden=True)
            )
        return self._true_neuromodulation_evaluator_hidden(
            W1, W2, receptor_densities, base_gains, neurotransmitter, inputs, targets,
            fitness_mode, modulation_mode, modulation_strength_override)

    def _eval_single_network_true_neuromodulated(
        self,
        W1_single: jnp.ndarray,
        W2_single: jnp.ndarray,
        receptor_densities_single: jnp.ndarray,
        base_gains_single: jnp.ndarray,
        neurotransmitter: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
        modulation_strength_override: Optional[float] = None,
    ) -> jnp.ndarray:
        """HMR's name (its neuromodulation variant): one network's fitness under true
        neuromodulation, the population evaluator on a population of one. The arguments are HMR's,
        in its order."""
        return self._eval_population_true_neuromodulated(
            W1_single[None],
            W2_single[None],
            receptor_densities_single[None],
            base_gains_single[None],
            neurotransmitter,
            inputs,
            targets,
            fitness_mode,
            modulation_mode,
            modulation_strength_override,
        )[0]

    def _eval_single_network_true_neuromodulated_with_hidden(
        self,
        W1_single: jnp.ndarray,
        W2_single: jnp.ndarray,
        receptor_densities_single: jnp.ndarray,
        base_gains_single: jnp.ndarray,
        neurotransmitter: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
        modulation_strength_override: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """HMR's name: _eval_single_network_true_neuromodulated that also returns the network's
        hidden activity, (cases, positions)."""
        fitness, hidden = self._eval_population_true_neuromodulated_with_hidden(
            W1_single[None],
            W2_single[None],
            receptor_densities_single[None],
            base_gains_single[None],
            neurotransmitter,
            inputs,
            targets,
            fitness_mode,
            modulation_mode,
            modulation_strength_override,
        )
        return fitness[0], hidden[0]

    def _eval_single_network_true_neuromodulated_multitask(
        self,
        W1_single: jnp.ndarray,
        W2_single: jnp.ndarray,
        receptor_densities_single: jnp.ndarray,
        base_gains_single: jnp.ndarray,
        task_nts: jnp.ndarray,
        task_inputs_list: Sequence[jnp.ndarray],
        task_targets_list: Sequence[jnp.ndarray],
    ) -> jnp.ndarray:
        """HMR's name: one network's mean fitness over tasks, each task under its NT vector (a row
        of task_nts), with HMR's settings for it (MSE, full modulation)."""
        total = 0.0
        for nt, inputs, targets in zip(task_nts, task_inputs_list, task_targets_list):
            total = total + self._eval_single_network_true_neuromodulated(
                W1_single,
                W2_single,
                receptor_densities_single,
                base_gains_single,
                nt,
                inputs,
                targets,
            )
        return total / len(task_inputs_list)

    def evaluate_with_custom_nt(
        self, custom_nt_vector: Any, inputs: Any, targets: Any
    ) -> jnp.ndarray:
        """HMR's name: the population's fitness on (inputs, targets) under an NT vector, with its
        settings (MSE, full modulation, tanh), from the weights and receptors the last generation
        cached.

        Raises:
            ValueError: an NT vector of the wrong size.
        """
        nt = jnp.asarray(custom_nt_vector, dtype=jnp.float32)
        num_nt = self.neuromod_config.num_nt_types
        if nt.shape != (num_nt,):
            raise ValueError(f"NT vector shape {nt.shape} != expected ({num_nt},)")
        return self.evaluate_with_neurotransmitter(nt, inputs, targets)

    def evaluate_multi_task_true_neuromodulated(self, tasks: Dict[str, Tuple[Any, Any]],
                                                aggregation: str = 'mean') -> jnp.ndarray:
        """HMR's: the population's aggregated fitness over tasks by name, each under its NT vector
        by name, from the cached weights; no selection. 'product' is the plain product, as in HMR.

        Raises:
            ValueError: an aggregation other than 'mean', 'min' or 'product'.
        """
        stacked = jnp.stack([self.evaluate_with_neurotransmitter(name, inputs, targets)
                             for name, (inputs, targets) in tasks.items()], axis=0)
        if aggregation == 'mean':
            return jnp.mean(stacked, axis=0)
        if aggregation == 'min':
            return jnp.min(stacked, axis=0)
        if aggregation == 'product':
            return jnp.prod(stacked, axis=0)
        raise ValueError(f"aggregation must be 'mean', 'min' or 'product', got {aggregation!r}")

    def set_task_neurotransmitter(self, task_idx: int, nt_vector: jnp.ndarray) -> None:
        """Set the neurotransmitter vector for a specific task.

        Args:
            task_idx: Task index
            nt_vector: NT vector (typically 4D: [DA, 5HT, NE, ACh])
        """
        self._multitask_state['current_task_idx'] = task_idx
        self._multitask_state['current_nt_vector'] = nt_vector

        if self.neuromod_config is not None:
            self.neuromod_config.neurotransmitter = nt_vector

    def get_task_neurotransmitter(self, task_name: str) -> jnp.ndarray:
        """Get the preset neurotransmitter vector for a named task.

        Args:
            task_name: One of 'xor', 'and', 'or', 'nand', 'nor', 'classification',
                      'regression', 'memory', 'sequential'

        Returns:
            NT vector for the task
        """
        return NT_TASK_PRESETS.get(task_name, NT_TASK_PRESETS['xor'])

    def evaluate_with_neurotransmitter(
        self,
        nt_vector: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        activation_name: Optional[str] = None,
        fitness_mode: str = 'mse',
        modulation_mode: str = 'full',
        branch_gating_mode: str = 'none',
    ) -> jnp.ndarray:
        """Evaluate population with a specific neurotransmitter vector and activation.

        This method allows evaluating the current population on arbitrary data
        with a custom NT vector and activation function. Useful for:
        - Testing networks on novel tasks without re-evolution
        - Probing multi-task capabilities
        - Benchmarking with different activation functions

        Prerequisites:
        - Must have called run_generation_verbose() or run_generation_multitask()
          at least once to cache W1, W2, receptor_densities, base_gains

        Args:
            nt_vector: Neurotransmitter vector, shape (num_nt,) typically (4,)
            inputs: Input data, shape (num_cases, num_inputs)
            targets: Target data, shape (num_cases, num_outputs)
            activation_name: Optional activation function name from ACTIVATION_FUNCTIONS.
                If None, uses 'tanh'. Examples: 'sin', 'relu', 'burst', 'gauss'
            fitness_mode: Fitness computation mode ('mse', 'accuracy', 'bce', etc.)
            modulation_mode: Neuromodulation mode ('full', 'gating_only', 'gain_bias_only')
            branch_gating_mode: 'none', 'spatial' or 'hierarchical' (multitask.branch_gating_mode):
                the routing a branch-gated multi-task generation applied; not read from the
                configuration.

        Returns:
            Fitness array of shape (pop_size,)

        Raises:
            RuntimeError: If weight matrices not cached (run_generation_verbose first)

        A task name in place of the vector stands for that task's preset NT vector
        (get_nt_for_task), as in HMR's version of this method.

        Example:
            >>> algo.run_generation_verbose(state, xor_problem)  # Cache weights
            >>> # Test with sin activation on a new task
            >>> fitness = algo.evaluate_with_neurotransmitter(
            ...     nt_vector=jnp.array([0.9, 0.1, 0.9, 1.0]),
            ...     inputs=jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            ...     targets=jnp.array([[0], [1], [1], [0]]),
            ...     activation_name='sin'
            ... )
        """
        if isinstance(nt_vector, str):
            nt_vector = get_nt_for_task(nt_vector, self.neuromod_config.num_nt_types)
        # Validate cached data exists
        if not hasattr(self, '_cached_W1') or self._cached_W1 is None:
            raise RuntimeError(
                "Weight matrices not cached. Call run_generation_verbose() or "
                "run_generation_multitask() first to build and cache W1, W2."
            )
        if not hasattr(self, '_neuromod_true') or self._neuromod_true is None:
            raise RuntimeError(
                "TRUE neuromodulation data not available. Ensure neuromodulation "
                "is enabled with mode='true_neuromodulation'."
            )

        W1 = self._cached_W1
        W2 = self._cached_W2
        rd = self._neuromod_true['receptor_densities']
        bg = self._neuromod_true['base_gains']

        # Check for H->H connections
        sparse_hh = getattr(self, '_cached_sparse_hh', None)
        has_hh = (
            sparse_hh is not None and
            self._forward_mode == ForwardPassMode.HYBRID_SPARSE_HH
        )
        activate_time = self.activate_time if has_hh else 0

        # Get activation function
        if activation_name is not None and activation_name not in ACTIVATION_FUNCTIONS:
            # An unknown name raises rather than running tanh
            raise ValueError(f"unknown activation {activation_name!r}; the names are "
                             f"{sorted(ACTIVATION_FUNCTIONS)}")
        act_fn = ACTIVATION_FUNCTIONS[activation_name] if activation_name else jnp.tanh

        # Create evaluation function
        check_branch_gating_and_penalty(
            branch_gating_mode, 0.0, num_nt=self.neuromod_config.num_nt_types
        )
        refuse_anti_branch_gating(self.neuromod_config.node_rule, branch_gating_mode)
        eval_population = self._create_eval_population_true_neuromodulated(
            has_hh=has_hh,
            activate_time=activate_time,
            hidden_act_fn=act_fn,
            branch_mode=branch_gating_mode,
            positions=(None if branch_gating_mode == 'none'
                       else get_hierarchical_grid(self.max_depth).all_positions),
        )

        # Ensure arrays are JAX arrays
        nt_vector = jnp.array(nt_vector)
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)

        # Evaluate
        if has_hh and sparse_hh is not None:
            fitness = eval_population(
                W1, W2, rd, bg, nt_vector, inputs, targets,
                sparse_hh.from_indices,
                sparse_hh.to_indices,
                sparse_hh.weights,
                sparse_hh.valid_mask,
                fitness_mode, modulation_mode, None
            )
        else:
            fitness = eval_population(
                W1, W2, rd, bg, nt_vector, inputs, targets,
                fitness_mode, modulation_mode, None
            )

        return fitness
