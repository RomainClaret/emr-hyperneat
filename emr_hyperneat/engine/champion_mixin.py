"""The champion and the population outside a generation: the substrate bridge, the best genome's
functions and metrics, and the continual-learning methods (freezing, injection, evaluation on new
problems and data)."""
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid
from emr_hyperneat.engine.evaluation import _eval_single_fitness
from emr_hyperneat.engine.functions import (
    ACTIVATION_FUNCTIONS,
    ACTIVATION_LIST,
    AGGREGATION_FUNCTIONS,
    AGGREGATION_LIST,
)
from emr_hyperneat.engine.metrics import EMRRecurrenceMetrics
from emr_hyperneat.execution import ExecutionMode


class ChampionMixin:
    """EMRHyperNEAT's champion, substrate and continual-learning methods."""

    def substrate_config(self) -> Any:
        """The substrate configuration this model's generation loop uses.

        Read from the parsed configuration, so defaults and the loop's own choices
        (e.g. the plain tanh/sigmoid evaluation when neither per-node mode is on)
        are the ones EMR applies. See `emr_hyperneat.substrate`.

        Returns:
            An `emr_hyperneat.substrate.SubstrateConfig`.

        Raises:
            RuntimeError: create_config has not been called.
            ValueError: the configuration evaluates something the substrate API does not
                describe (neuromodulation, multi-task, swarm mode, LEO, the weight-interpretation or
                random function modes, a non-standard execution mode, or hidden-to-hidden
                connections other than per-genome discovery without the h->h cache).
        """
        # Lazy import: emr_hyperneat.substrate imports emrhyperneat, which imports this module.
        from emr_hyperneat.substrate import SubstrateConfig

        if self.max_depth is None or not hasattr(self, 'dynamic_functions_mode'):
            raise RuntimeError("call create_config before substrate_config")

        refusals = []
        if self.network_model == 'node_list':
            # The node-list model's networks live on a quadtree's leaves
            refusals.append("recurrence.network_model 'node_list' (the API describes the grid "
                            "model)")
        if self.dynamic_functions_mode not in ('disabled', 'global', 'cppn_output'):
            refusals.append(f"dynamic_functions.mode={self.dynamic_functions_mode!r}")
        if self.agg_mode not in ('disabled', 'global', 'cppn_output'):
            refusals.append(f"aggregation.mode={self.agg_mode!r}")
        recurrent = self.extended_config is not None and self.extended_config.allow_hidden_to_hidden
        if recurrent:
            # The API builds hidden-to-hidden connections as per-genome discovery does; what it
            # cannot reproduce is refused.
            ec = self.extended_config
            if ec.hh_discovery_mode != 'per_genome':
                refusals.append(f"hh_discovery_mode {ec.hh_discovery_mode!r} (the API builds per_genome discovery)")
            if ec.use_dense_discovery:
                refusals.append("use_dense_discovery")
            if ec.hh_cache_enabled:
                refusals.append("the h->h cache (it reuses connections discovered for other "
                                "genomes)")
            if self.hh_aggregation_mode == 'dynamic' and not (
                    self.agg_mode == 'cppn_output' and self.num_cppn_outputs > 1):
                refusals.append("hh_aggregation_mode 'dynamic' without the aggregation channel "
                                "(the weight-interpretation decode)")
        if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
            refusals.append(f"neuromodulation mode {self.neuromod_config.mode!r}")
        if self.multitask_config is not None and getattr(self.multitask_config, 'enabled', False):
            refusals.append("multi-task evaluation")
        if self.swarm_mode:
            refusals.append("swarm mode")
        if getattr(self, 'use_leo', False):
            refusals.append("LEO gating")
        if self.agg_palette is not None:
            refusals.append("an aggregation palette")
        if getattr(self, 'df_palette_evolution', None) is not None:
            refusals.append("palette evolution (its decode)")
        if (self.df_cppn_read, self.agg_cppn_read) != ('self', 'self') or 'raw' in (
            self.df_weight_source,
            self.agg_weight_source,
        ):
            refusals.append("HMR's reads (cppn_read or weight_source)")
        resolved = self.execution_config.resolve(self.max_depth)
        if resolved is not ExecutionMode.STANDARD:
            refusals.append(f"execution mode {resolved.value!r} (set execution.mode: standard)")
        if refusals:
            raise ValueError(
                "substrate_config describes the substrate of the standard "
                "execution path; this configuration also uses " + ", ".join(refusals)
            )

        # The generation loop's plain branch runs when neither per-node mode is on, and
        # it applies tanh and sigmoid whatever output_activation says.
        if self.dynamic_functions_mode == 'disabled' and self.agg_mode == 'disabled':
            hidden, output = 'tanh', 'sigmoid'
        else:
            name = self.df_hidden_activation if self.dynamic_functions_mode == 'global' else 'tanh'
            hidden = name if name in ACTIVATION_FUNCTIONS else 'tanh'
            output = (self.df_output_activation
                      if self.df_output_activation in ACTIVATION_FUNCTIONS else 'sigmoid')
        global_agg = (self.agg_global_function
                      if self.agg_global_function in AGGREGATION_FUNCTIONS else 'sum')

        return SubstrateConfig(
            input_coords=tuple(tuple(float(c) for c in xy) for xy in self.substrate_input_coords),
            output_coords=tuple(tuple(float(c) for c in xy) for xy in self.substrate_output_coords),
            max_depth=int(self.max_depth),
            variance_threshold=float(self.variance_threshold),
            initial_depth=int(self.initial_depth),
            variance_rule=self.variance_rule,
            max_weight=float(self.max_weight),
            weight_threshold=self.weight_threshold,
            locality_radius=self.locality_radius,
            geometry_seeding=bool(self.geometry_seeding_enabled),
            num_cppn_outputs=int(self.num_cppn_outputs),
            activation_mode=self.dynamic_functions_mode,
            activation_channel=int(self._cppn_output_indices['activation']),
            num_activations=int(self.df_num_activations),
            palette=(tuple(int(i) for i in np.asarray(self.df_palette))
                     if self.df_palette is not None else None),
            hidden_activation=hidden,
            output_activation=output,
            aggregation_mode=self.agg_mode,
            aggregation_channel=int(self._cppn_output_indices['aggregation']),
            num_aggregations=int(self.agg_num_aggregations),
            global_aggregation=global_agg,
            true_aggregation=bool(self.agg_use_true_aggregation),
            **self._substrate_recurrence_fields(recurrent),
            activation_index_decode=self.df_index_decode,
            aggregation_index_decode=self.agg_index_decode,
        ).validate()

    def _substrate_recurrence_fields(self, recurrent: bool) -> Dict[str, Any]:
        """The SubstrateConfig recurrence fields of this configuration (none when feed-forward)."""
        if not recurrent:
            return {}
        ec = self.extended_config
        return dict(
            hidden_to_hidden=True,
            allow_backward=bool(ec.allow_backward),
            allow_lateral=bool(ec.allow_lateral),
            allow_self_loops=bool(ec.allow_self_loops),
            band_threshold=float(self.band_threshold),
            iteration_level=int(ec.iteration_level),
            hop_decay_factor=float(ec.hop_decay_factor),
            multi_hop_algorithm=ec.multi_hop_algorithm,
            max_sparse_conns=int(ec.max_sparse_conns),
            sparse_cap_policy=ec.sparse_cap_policy,
            max_sparse_conns_ceiling=ec.max_sparse_conns_ceiling,
            activate_time=int(self.activate_time),
            settling_rule=self._settling_rule,
            hh_activation_mode=self.hh_activation_mode,
            hh_aggregation_mode=self.hh_aggregation_mode,
        )

    def population_substrates(self, state: Any, device_id: int = 0) -> Any:
        """The substrates of the population in ``state``, as the next generation builds them.

        With recurrence the hidden-to-hidden query keeps every genome's pair outputs until the connections are
        built: ``count**2 * num_cppn_outputs`` float32 values per genome for ``count`` active positions (8 MB per
        genome at 1,000 positions and two outputs), where the generation loop holds one bucket slice at a time.

        Args:
            state: Algorithm state from initialize or run_generation.
            device_id: Device for the feed-forward query; the hidden-to-hidden query runs on the default device.

        Returns:
            An `emr_hyperneat.substrate.Substrate` with one entry per genome.
        """
        from emr_hyperneat.substrate import (
            build_recurrent_substrate,
            build_substrate,
            query_recurrent_substrate,
            query_substrate,
        )

        config = self.substrate_config()
        population = self._compiled_ask(state)
        transformed = self._compiled_transform_batch(state, population)
        substrate = build_substrate(
            *query_substrate(self._jitted_cppn_forward, state, transformed, config,
                             device_id=device_id),
            config,
        )
        if config.hidden_to_hidden:
            # The public two-step build, with this model's CPPNs as the evaluator
            substrate = build_recurrent_substrate(
                substrate, query_recurrent_substrate(self._jitted_cppn_forward, state, transformed, substrate))
        return substrate

    def get_extended_metrics(self) -> Optional[EMRRecurrenceMetrics]:
        """Get extended metrics from last generation."""
        return self._extended_metrics

    def get_best_activation_indices(self) -> Dict[str, Any]:
        """Function histograms over the hidden nodes of the last generation's champion.

        The champion is the one :meth:`evaluate_on_data` scores, and the histograms count the
        same nodes as its ``activation_counts``: a position the mask drops computes nothing and
        is not counted, though its raw CPPN output of 0 would decode to the palette's middle
        entry. A palette entry is named by its ``ACTIVATION_LIST`` position, not by its rank in
        the palette.

        Returns:
            An empty dict before the first generation and after the paths that record no
            champion (depth offload, multi-task generations, the multi-GPU strategies).
            Otherwise 'num_positions' (positions on the evaluation axis) and
            'num_hidden_nodes', plus 'activation_histogram' and 'aggregation_histogram'
            (function name -> number of hidden nodes) where functions are assigned per node.
        """
        record = self._last_eval
        if record is None:
            return {}
        mask = np.asarray(record['mask_eval'], dtype=bool)
        result = {'num_positions': int(mask.shape[0]), 'num_hidden_nodes': int(mask.sum())}

        def histogram(indices, names):
            values, numbers = np.unique(np.asarray(indices)[mask], return_counts=True)
            return {names[int(v)]: int(n) for v, n in zip(values, numbers)}

        if record['act_indices'] is not None:
            result['activation_histogram'] = histogram(record['act_indices'], ACTIVATION_LIST)
        if record['agg_indices'] is not None:
            result['aggregation_histogram'] = histogram(record['agg_indices'], AGGREGATION_LIST)
        return result

    def evaluate_best_all_modes(self) -> Optional[Dict[str, float]]:
        """All six fitness metrics of the last generation's champion on the training data.

        The champion's outputs come from the forward it was evaluated with (see
        :meth:`evaluate_on_data`), so every metric describes the evolved network. Must be
        called after a generation.
        """
        record = self._last_eval
        if record is None:
            return None
        inputs = self._cached_inputs
        targets = self._cached_targets
        forward_one = record['forward_one']
        modes = ('mse', 'accuracy', 'hybrid', 'bce', 'soft_accuracy')

        def all_modes(*args):
            outputs = forward_one(inputs, *args)
            return [_eval_single_fitness(outputs, targets, mode) for mode in modes]

        # Inside the vmap over the champion's two copies, as the generation scored it
        values = jax.vmap(all_modes)(*record['champion_args'])
        scored = {mode: float(value[0]) for mode, value in zip(modes, values)}
        # acc_mse is not an _eval_single_fitness mode: accuracy plus a small MSE tiebreak
        scored['acc_mse'] = scored['accuracy'] + 0.01 * scored['mse']
        return {mode: scored[mode]
                for mode in ('mse', 'accuracy', 'acc_mse', 'hybrid', 'bce', 'soft_accuracy')}

    #: The structural mutations a topology freeze stops (TensorNEAT's DefaultMutation rates).
    STRUCTURAL_MUTATIONS = ('node_add', 'node_delete', 'conn_add', 'conn_delete')

    def get_best_genome_with_fitness(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """The last evaluated generation's champion, as run_generation recorded it.

        HMR's version took the argmax of a fitness array over the population in the state it was
        given, which after a generation is the next population (TensorNEAT lays it out best species
        first), so it returned another genome unless the argmax was 0; it had no caller. This one
        takes no arguments: the record knows the genome.

        Returns:
            ``(nodes, conns, fitness, index)``: the champion's CPPN genes, its fitness and its index
            in the evaluated population (the next population keeps it, unmutated, first).

        Raises:
            RuntimeError: no generation has been evaluated (only the per-generation path records a
                champion).
        """
        record = getattr(self, '_last_eval', None)
        if record is None:
            raise RuntimeError(
                "no evaluated generation recorded a champion: run run_generation first (of "
                "run_until_threshold's multi-device routes, only the split routes and the island "
                "model record one)"
            )
        nodes, conns = record['champion_genome']
        return nodes, conns, record['champion_fitness'], record['champion_index']

    def inject_genome_into_population(
        self,
        state: Any,
        nodes: Any,
        conns: Any,
        num_copies: int = 1,
        slots: Optional[Sequence[int]] = None,
    ) -> Any:
        """Put copies of a genome into the population in ``state``, the genomes the next generation
        evaluates.

        The copies replace the genomes at ``slots``: by default the last ``num_copies`` slots,
        offspring. After a generation slot 0 holds the champion, which TensorNEAT keeps unmutated,
        and HMR's version wrote slots 0 to num_copies - 1, so in an evolved population it replaced
        the champion; ``slots=range(num_copies)`` reproduces HMR's slots. Species bookkeeping is
        left to the next generation's speciation, as in HMR: for that generation the copies count in
        the replaced genomes' species (harmless in a fresh population, which has one species).

        Args:
            state: Algorithm state (from initialize or run_generation).
            nodes, conns: The genome's CPPN genes, shaped as one row of the population's.
            num_copies: Copies, when ``slots`` is not given.
            slots: Population indices to overwrite.

        Returns:
            The state with the copies in place.

        Raises:
            ValueError: genes shaped unlike the population's, other input or output node keys, slots
                out of range or repeated, or the h->h cache on (it reuses connections by slot).
        """
        if self.extended_config is not None and self.extended_config.hh_cache_enabled:
            raise ValueError(
                "the h->h cache reuses connections by population slot, so an injected genome would "
                "be scored with the replaced genome's connections: turn "
                "recurrence.hh_cache_enabled off"
            )
        pop_nodes, pop_conns = state.pop_nodes, state.pop_conns
        pop_size = int(pop_nodes.shape[0])
        nodes = jnp.asarray(nodes, dtype=pop_nodes.dtype)
        conns = jnp.asarray(conns, dtype=pop_conns.dtype)
        if nodes.shape != pop_nodes.shape[1:] or conns.shape != pop_conns.shape[1:]:
            raise ValueError(
                f"the genome's genes are shaped {nodes.shape} and {conns.shape}; the population's "
                f"{pop_nodes.shape[1:]} and {pop_conns.shape[1:]}"
            )
        io = len(self.neat_algo.genome.input_idx) + len(self.neat_algo.genome.output_idx)
        if not np.array_equal(np.asarray(nodes[:io, 0]), np.asarray(pop_nodes[0, :io, 0])):
            raise ValueError(
                "the genome's input and output node keys differ from the population's (a CPPN of "
                "another input or output count)"
            )
        if slots is None:
            if not 1 <= num_copies <= pop_size:
                raise ValueError(
                    f"num_copies must be between 1 and the population size {pop_size}, got "
                    f"{num_copies}"
                )
            slots = range(pop_size - num_copies, pop_size)
        slots = [int(s) for s in slots]
        if len(set(slots)) != len(slots) or any(not 0 <= s < pop_size for s in slots):
            raise ValueError(f"slots must be distinct indices below {pop_size}, got {slots}")
        index = jnp.asarray(slots, dtype=jnp.int32)
        return state.update(
            pop_nodes=pop_nodes.at[index].set(jnp.broadcast_to(nodes, (len(slots),) + nodes.shape)),
            pop_conns=pop_conns.at[index].set(jnp.broadcast_to(conns, (len(slots),) + conns.shape)),
        )

    def freeze_topology(self, state: Any = None) -> Any:
        """Stop structural mutation: node and connection additions and deletions.

        Crossover, weight, bias and response mutation and the replacement of a CPPN node's functions
        continue, as in HMR, so selection still chooses among the population's existing topologies
        and no new one appears. The rates are read when TensorNEAT's tell is traced, so tell is
        compiled again (from a new function object, so no trace cache of the old rates is reused).
        The freeze is the model's: a new create_config ends it.

        Args:
            state: Accepted for HMR's signature, which stored a flag in the state; returned
                unchanged.

        Returns:
            ``state``.
        """
        mutation = self.neat_algo.genome.mutation
        if getattr(self, '_frozen_structural_rates', None) is None:
            self._frozen_structural_rates = {
                name: getattr(mutation, name) for name in self.STRUCTURAL_MUTATIONS
            }
        for name in self.STRUCTURAL_MUTATIONS:
            setattr(mutation, name, 0.0)
        self._recompile_tell()
        return state

    def unfreeze_topology(self, state: Any = None) -> Any:
        """Restore the structural mutation rates a freeze stopped (and compile tell again).

        Args:
            state: Returned unchanged.

        Returns:
            ``state``.
        """
        rates = getattr(self, '_frozen_structural_rates', None)
        if rates is not None:
            mutation = self.neat_algo.genome.mutation
            for name, rate in rates.items():
                setattr(mutation, name, rate)
            self._frozen_structural_rates = None
            self._recompile_tell()
        return state

    def is_topology_frozen(self, state: Any = None) -> bool:
        """Whether structural mutation is stopped (the model's freeze, checked against TensorNEAT's
        rates).

        Raises:
            RuntimeError: the freeze and the mutation rates disagree (rates changed behind the API).
        """
        frozen = getattr(self, '_frozen_structural_rates', None) is not None
        mutation = self.neat_algo.genome.mutation
        zero = all(getattr(mutation, name) == 0 for name in self.STRUCTURAL_MUTATIONS)
        if frozen and not zero:
            raise RuntimeError("the topology freeze is recorded but a structural mutation rate is "
                               "not zero")
        return frozen

    def _recompile_tell(self) -> None:
        neat_algo = self.neat_algo
        self._compiled_tell = jax.jit(lambda state, fitness: neat_algo.tell(state, fitness))
        # A compiled generation loop holds the previous tell
        self._generation_loop_cache = {}

    def evaluate_genome_on_problem(
        self,
        state: Any,
        nodes: Any,
        conns: Any,
        problem: Any = None,
        inputs: Any = None,
        targets: Any = None,
        input_coords: Optional[Sequence[Sequence[float]]] = None,
        output_coords: Optional[Sequence[Sequence[float]]] = None,
    ) -> float:
        """A genome's fitness on a problem's data, as this model's generation loop would score it.

        The genome's substrate is built by the substrate API from its CPPN (variance masks, weights,
        per-node functions, hidden-to-hidden connections) and run with this model's fitness mode. On
        the training problem it reproduces the champion's recorded fitness (bit for bit feed-forward
        on the CPU; within float32 rounding with recurrence, which the loop evaluates on the
        population's union of positions). HMR's evaluator took every position as a hidden node (no
        variance mask), ignored per-node functions, recurrence, LEO and the fitness mode, and with a
        multi-output CPPN interleaved the channels, so it did not reproduce the training fitness
        even on the training problem.

        Args:
            state: Algorithm state (TensorNEAT's, for the CPPN forward).
            nodes, conns: The genome's CPPN genes.
            problem: A problem with ``get_data()`` or ``inputs`` and ``targets``; or pass ``inputs``
                and ``targets``.
            inputs, targets: The data, (samples, inputs) and (samples, outputs).
            input_coords, output_coords: Substrate coordinates for a problem whose input or output
                count differs from this model's.

        Returns:
            The fitness.

        Raises:
            ValueError: a configuration the substrate API does not describe (neuromodulation,
                multi-task, ...), or data that does not match the coordinates.
        """
        from emr_hyperneat.substrate import (
            build_recurrent_substrate,
            build_substrate,
            query_recurrent_substrate,
            query_substrate,
            substrate_forward,
        )
        if problem is not None:
            if hasattr(problem, 'get_data'):
                data = problem.get_data()
                inputs = [d[0] for d in data]
                targets = [d[1] for d in data]
            else:
                inputs, targets = problem.inputs, problem.targets
        if inputs is None or targets is None:
            raise ValueError("pass a problem or inputs and targets")
        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        targets = jnp.asarray(targets, dtype=jnp.float32)
        if targets.ndim == 1:
            targets = targets[:, None]
        config = self.substrate_config()
        if input_coords is not None:
            config = config._replace(
                input_coords=tuple(tuple(float(c) for c in xy) for xy in input_coords)
            )
        if output_coords is not None:
            config = config._replace(
                output_coords=tuple(tuple(float(c) for c in xy) for xy in output_coords)
            )
        if inputs.shape[1] != len(config.input_coords) or targets.shape[1] != len(
            config.output_coords
        ):
            raise ValueError(
                f"the data has {inputs.shape[1]} inputs and {targets.shape[1]} outputs; the "
                f"substrate {len(config.input_coords)} and {len(config.output_coords)}: pass "
                "input_coords and output_coords for this problem"
            )
        # Two copies: a batch of one lowers to other reduction kernels (see _record_evaluation)
        population = (jnp.stack([jnp.asarray(nodes)] * 2), jnp.stack([jnp.asarray(conns)] * 2))
        transformed = self._compiled_transform_batch(state, population)
        substrate = build_substrate(
            *query_substrate(self._jitted_cppn_forward, state, transformed, config), config
        )
        if config.hidden_to_hidden:
            substrate = build_recurrent_substrate(
                substrate,
                query_recurrent_substrate(self._jitted_cppn_forward, state, transformed, substrate),
            )
        outputs = substrate_forward(substrate, inputs)
        fitness = jax.vmap(lambda out: _eval_single_fitness(out, targets, self.fitness_mode))(
            outputs
        )
        return float(fitness[0])

    def create_state_for_new_problem(
        self,
        problem: Any,
        seed: int = 42,
        inherit_genome: Optional[Tuple[Any, Any]] = None,
        num_copies: int = 1,
        config_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """A fresh population for another problem, optionally seeded with a genome.

        With ``config_params`` the model is configured again first (for a problem whose input or
        output count differs, the parameters carry its substrate coordinates); otherwise the current
        configuration is kept. HMR's version rebuilt a configuration from six settings, so every
        other setting reverted to its default, and raised when no palette was set; it had no caller.

        Args:
            problem: The new problem.
            seed: The new population's seed.
            inherit_genome: ``(nodes, conns)`` to inject, or None.
            num_copies: Copies of the inherited genome (in the last slots, see
                inject_genome_into_population).
            config_params: Parameters for create_config, or None to keep the current configuration.

        Returns:
            The new state.
        """
        if config_params is not None:
            self.create_config(config_params)
        state = self.initialize(self.neat_algo, problem, seed=seed)
        if inherit_genome is not None:
            state = self.inject_genome_into_population(
                state, *inherit_genome, num_copies=num_copies
            )
        return state

    # Test Evaluation with Dynamic Function Support

    def evaluate_on_data(
        self,
        state: Any,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate the last generation's champion on arbitrary input/target data.

        The champion is scored with the forward its generation evaluated it with: the same
        per-genome function, arrays and fitness mode, rescaling included (see
        ``_build_forward``). On the training data its fitness is therefore the fitness it was
        selected on (bit for bit on CPU), and a held-out score describes the network that was
        evolved, recurrent connections, per-node aggregation, global and random activation
        modes and neuromodulation included, not a dense tanh network rebuilt from it.

        The champion is recorded by :meth:`run_generation_verbose`, the generation of the
        standard and streaming execution modes on one task, which the split routes
        (POPULATION_PMAP, POSITION_SHARDING, HYBRID) also run. Depth offload, multi-task
        generations and the other multi-device routes record none, and this raises after them.

        Args:
            state: The state the last generation returned. Its first genome is that
                generation's champion: TensorNEAT places the best genome, unmutated, first in
                the next population, and this is checked.
            inputs: Input data array of shape (n_samples, n_features)
            targets: Target data array of shape (n_samples, n_outputs)

        Returns:
            Dict with 'mse', 'fitness', 'n_samples', the error distribution, substrate
            statistics, 'champion_index', 'generation' and 'training_fitness', and, when a
            dynamic-functions mode is on, 'activation_counts' and
            'activation_counts_per_level' over the champion's hidden nodes.

        Raises:
            RuntimeError: The last generation recorded no champion.
            ValueError: ``state`` is not the one the last generation returned.
        """
        record = self._last_eval
        if record is None:
            raise RuntimeError(
                "evaluate_on_data scores the champion of the last generation this model ran "
                "with run_generation_verbose (standard or streaming execution, one task); run "
                "a generation first. Depth offload, multi-task generations, the multi-GPU "
                "strategies and execution.generation_loop 'while_loop' record no champion"
            )
        population = self._compiled_ask(state)
        champion_nodes, champion_conns = record['champion_genome']
        if not (np.array_equal(np.asarray(population[0][0]), champion_nodes, equal_nan=True)
                and np.array_equal(np.asarray(population[1][0]), champion_conns,
                                   equal_nan=True)):
            raise ValueError(
                "evaluate_on_data scores the champion of the last generation this model ran, "
                "and the state passed is not the one that generation returned"
            )

        inputs = jnp.asarray(inputs, dtype=jnp.float32)
        targets = jnp.asarray(targets, dtype=jnp.float32)
        gain = (
            getattr(self.neuromod_config, 'modulatory_matched_gain', 'none')
            if self.neuromod_config
            else 'none'
        )
        if (self.neuromod_config is not None and self.neuromod_config.mode == 'modulatory_neurons'
                and gain != 'none' and inputs.shape[0] < 2):
            # The matched-gain controls act over the cases evaluated together: one sample would
            # score 'none'
            raise ValueError(
                f"neuromodulation.modulatory_matched_gain {gain!r} acts over the cases evaluated "
                f"together, and one sample makes it equal 'none'; evaluate two or more"
            )
        forward_one = record['forward_one']
        fitness_mode = self.fitness_mode

        def score(*args):
            outputs = forward_one(inputs, *args)
            return outputs, _eval_single_fitness(outputs, targets, fitness_mode)

        # vmapped over the champion's two copies, as the generation vmapped the population
        outputs, fitness = jax.vmap(score)(*record['champion_args'])
        if self._fitness_rescale_variance is not None:
            fitness = jnp.maximum(
                0.0, 1.0 - (1.0 - fitness) / self._fitness_rescale_variance
            )
        outputs = outputs[0]
        errors = (outputs - targets) ** 2
        sample_errors = jnp.mean(errors, axis=1)

        # The node-list model's champion lives on the quadtree's leaves, not on the grid, and
        # also carries leaf-to-leaf weights
        node_list_record = 'W_hh_full' in record
        h_grid = None if node_list_record else get_hierarchical_grid(self.max_depth)
        mask_full = record['mask_full']
        W1_best, W2_best = record['W1_full'], record['W2_full']
        W1_nonzero = int(jnp.sum(jnp.abs(W1_best) > 0))
        W2_nonzero = int(jnp.sum(jnp.abs(W2_best) > 0))
        hidden_nonzero = int(jnp.sum(jnp.abs(record['W_hh_full']) > 0)) if node_list_record else 0
        W1_flat, W2_flat = W1_best.flatten(), W2_best.flatten()
        W1_active = W1_flat[jnp.abs(W1_flat) > 0]
        W2_active = W2_flat[jnp.abs(W2_flat) > 0]
        result: Dict[str, Any] = {
            'mse': float(jnp.mean(errors)),
            'fitness': float(fitness[0]),
            'n_samples': int(inputs.shape[0]),
            'mse_std': float(jnp.std(sample_errors)),
            'mse_min': float(jnp.min(sample_errors)),
            'mse_max': float(jnp.max(sample_errors)),
            'mse_median': float(jnp.median(sample_errors)),
            'active_hidden_nodes': int(np.sum(mask_full)),
            'total_positions': (int(mask_full.shape[0]) if node_list_record
                                else h_grid.total_positions),
            'total_connections': W1_nonzero + W2_nonzero + hidden_nonzero,
            'input_connections': W1_nonzero,
            'output_connections': W2_nonzero,
            'w1_mean': float(jnp.mean(W1_active)) if W1_active.size else 0.0,
            'w1_std': float(jnp.std(W1_active)) if W1_active.size else 0.0,
            'w2_mean': float(jnp.mean(W2_active)) if W2_active.size else 0.0,
            'w2_std': float(jnp.std(W2_active)) if W2_active.size else 0.0,
            '_mask': mask_full,
            '_W1': W1_best,
            '_W2': W2_best,
            '_h_grid': h_grid,
            'champion_index': record['champion_index'],
            'generation': record['generation'],
            'training_fitness': record['champion_fitness'],
            'dynamic_functions_mode': self.dynamic_functions_mode,
        }
        if node_list_record:
            result['hidden_connections'] = hidden_nonzero
            result['_W_hh'] = record['W_hh_full']
            result['network_model'] = 'node_list'
        if self.dynamic_functions_mode != 'disabled':
            result.update(self._champion_activation_counts(record, h_grid))
        return result

    def _champion_activation_counts(self, record: Dict[str, Any], h_grid: Any) -> Dict[str, Any]:
        """How many of the champion's hidden nodes use each activation, overall and per level.

        Hidden nodes only: positions the mask drops compute nothing, and counting them, decoded
        at a raw output of 0, would inflate the middle function.
        """
        mask = np.asarray(record['mask_eval'], dtype=bool)
        positions = (record['eval_positions'] if record['eval_positions'] is not None
                     else np.arange(mask.shape[0]))
        if record['act_indices'] is not None:
            names = np.array([ACTIVATION_LIST[i] for i in np.asarray(record['act_indices'])])
        else:
            one = (self.df_hidden_activation if self.dynamic_functions_mode == 'global'
                   else 'tanh')
            names = np.full(mask.shape[0], one)
        live_names, live_positions = names[mask], np.asarray(positions)[mask]

        def counts(selected):
            values, numbers = np.unique(selected, return_counts=True)
            return {str(v): int(n) for v, n in zip(values, numbers)}

        offsets = h_grid.level_offsets_static
        per_level = {}
        for level in range(len(offsets) - 1):
            in_level = (live_positions >= offsets[level]) & (live_positions < offsets[level + 1])
            per_level[f'level_{level}'] = {
                'num_positions': int(offsets[level + 1] - offsets[level]),
                'num_hidden_nodes': int(np.sum(in_level)),
                'counts': counts(live_names[in_level]),
            }
        overall = counts(live_names)
        return {
            'activation_counts': overall,
            'activation_counts_per_level': per_level,
            'num_activations_used': len(overall),
        }
