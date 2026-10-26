"""Execution modes and the alternative generations: the memory modes (offloading, sharding, chunk sizes),
the streaming build of the weight matrices, and the node-list model's generation."""
import time
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat import node_list
from emr_hyperneat._compat.core.base_algorithm import AlgorithmMetrics
from emr_hyperneat.emrhyperneat_base import HierarchicalGridStructure, MultiGPUStrategy
from emr_hyperneat.engine.evaluation import (
    _auto_eval_chunk_size,
    _eval_single_fitness,
    _node_list_forward,
    chunked_forward_eval,
)
from emr_hyperneat.engine.leo import LEO_CHANNEL, leo_sparsify
from emr_hyperneat.engine.metrics import (
    EMRRecurrenceMetrics,
    fitness_statistics,
    sanitize_fitness,
    species_summary,
    topology_summary,
)
from emr_hyperneat.engine.substrate_build import (
    _accumulate_level_slice,
    apply_locality_penalty_to_weights,
    compute_single_level_variance,
    legacy_level0_variance,
    query_level_positions,
)
from emr_hyperneat.execution import ExecutionMode
from emr_hyperneat.execution.chunking import (
    optimal_position_chunk_size,
    population_chunk_for_query,
    population_chunk_size,
)


class ModesMixin:
    """EMRHyperNEAT's execution modes and the node-list generation."""

    def _apply_execution_config(self) -> None:
        """Mirror `execution_config` onto the flat attributes the strategies read.

        The strategy functions and the existing tests both address these by name,
        so the config object and the imperative setters write the same place rather
        than competing.
        """
        config = self.execution_config
        self._depth_offload_threshold = config.depth_threshold
        self._eval_chunk_levels = config.eval_chunk_levels
        self._eval_chunk_positions = config.eval_chunk_positions
        self._cppn_pos_chunk_size = config.cppn_pos_chunk_size
        # `max_lookahead_positions` is a memory budget, not a truncation trigger:
        # a level larger than it is computed in slabs, which bounds memory without
        # changing the substrate. Exceeding it truncates (zeroed variance, subdivision
        # halted at and below that level) only when `truncate_variance_lookahead`
        # asks for it. A budget of float('inf') would not mean "never truncate": it
        # would disable the budget too, so the whole next level would be materialized,
        # which would run out of memory at depth 12 (67M positions per genome).
        self._max_lookahead_positions = config.max_lookahead_positions
        self._truncate_variance_lookahead = config.truncate_variance_lookahead
        self._skip_dead_levels = config.skip_dead_levels
        self._offload_verbose = config.verbose
        self._target_shard_gb = config.target_shard_gb
        self._target_eval_shard_gb = config.target_eval_shard_gb
        self._enable_disk_offload = config.enable_disk_offload
        self._disk_offload_path = config.disk_offload_path
        self._target_cpu_gb = config.target_cpu_gb
        self._max_population_chunk = config.max_population_chunk
        self._use_memmap = config.use_memmap
        self._memmap_path = config.memmap_path
        self._use_cpu_grid = config.use_cpu_grid
        self._recurrent_target_memory_gb = config.recurrent_target_memory_gb

    def configure_offloading(
        self,
        depth_threshold: int = 9,
        eval_chunk_levels: int = 3,
        eval_chunk_positions: int = 1_000_000,
        cppn_pos_chunk_size: int = 100_000,
        max_lookahead_positions: int = 5_000_000,
        verbose: bool = False,
        use_cpu_grid: bool = False,
        skip_dead_levels: Optional[bool] = None,
    ) -> None:
        """Configure depth offloading imperatively.

        Kept alongside the `execution:` config block so existing code and tests
        keep working; both write `execution_config`.

        `max_lookahead_positions` is a memory budget. Levels larger than it are
        computed in slabs, which bounds memory and leaves the substrate unchanged,
        so asking for a limit does not mean accepting a different substrate.
        Truncation (zeroed variance, and subdivision halted at and below that level)
        is available through `truncate_variance_lookahead` on the config, and is off
        by default.

        Args:
            depth_threshold: Depth at or above which `auto` selects offloading.
            eval_chunk_levels: Levels loaded at once during evaluation.
            eval_chunk_positions: Maximum positions per device chunk.
            cppn_pos_chunk_size: Maximum positions per CPPN query.
            max_lookahead_positions: Levels larger than this skip variance lookahead.
            verbose: Per-level statistics.
            use_cpu_grid: Keep the grid in host RAM, needed from depth 12.
            skip_dead_levels: Skip the evaluation of levels no genome reaches (HMR's keyword; the
                execution block's key of the same name). None keeps the configured value.
        """
        if skip_dead_levels is not None:
            if not isinstance(skip_dead_levels, bool):
                raise ValueError(f"skip_dead_levels must be a bool, got {skip_dead_levels!r}")
            self.execution_config.skip_dead_levels = skip_dead_levels
        self.execution_config.depth_threshold = depth_threshold
        self.execution_config.eval_chunk_levels = eval_chunk_levels
        self.execution_config.eval_chunk_positions = eval_chunk_positions
        self.execution_config.cppn_pos_chunk_size = cppn_pos_chunk_size
        self.execution_config.max_lookahead_positions = max_lookahead_positions
        self.execution_config.verbose = verbose
        self.execution_config.use_cpu_grid = use_cpu_grid
        self._apply_execution_config()

    def configure_sharding(
        self,
        target_shard_gb: float = 0.5,
        target_eval_shard_gb: float = 1.0,
        enable_disk_offload: bool = False,
        disk_offload_path: str = '/tmp/emr_offload',
        max_population_chunk: Optional[int] = None,
        target_cpu_gb: float = 32.0,
        use_memmap: bool = True,
        memmap_path: str = '/tmp/emr_memmap',
    ) -> None:
        """Configure intra-level sharding and host residency imperatively.

        Args:
            target_shard_gb: Device memory per position shard during CPPN queries.
            target_eval_shard_gb: Device memory per shard during evaluation.
            enable_disk_offload: Spill to disk at extreme depth.
            disk_offload_path: Directory for disk offloading.
            max_population_chunk: Override the computed population chunk size.
            target_cpu_gb: Host RAM budget for the weight matrices.
            use_memmap: Back large chunks with memory-mapped files.
            memmap_path: Directory for memmap files.
        """
        self.execution_config.target_shard_gb = target_shard_gb
        self.execution_config.target_eval_shard_gb = target_eval_shard_gb
        self.execution_config.enable_disk_offload = enable_disk_offload
        self.execution_config.disk_offload_path = disk_offload_path
        self.execution_config.max_population_chunk = max_population_chunk
        self.execution_config.target_cpu_gb = target_cpu_gb
        self.execution_config.use_memmap = use_memmap
        self.execution_config.memmap_path = memmap_path
        self._apply_execution_config()

    def resolve_execution_mode(self) -> ExecutionMode:
        """Resolve the configured mode against `max_depth`. Never returns AUTO.

        On a GPU backend the resolution is memory-aware: AUTO escalates a
        STANDARD selection to DEPTH_OFFLOAD when the standard path's calibrated
        working-set estimate exceeds the device (the d7/pop1000 wall the depth
        thresholds alone could not see). On CPU, and whenever sizing is
        unavailable, resolution is exactly the static threshold logic.
        """
        from emr_hyperneat.execution import (
            resolve_mode,
        )
        population = getattr(self, 'population_size', None)
        device_bytes = None
        if population is not None and jax.default_backend() == 'gpu':
            try:
                stats = jax.local_devices()[0].memory_stats() or {}
                device_bytes = stats.get('bytes_limit')
            except Exception:
                device_bytes = None
        coords_in = getattr(self, '_cached_input_coords', None)
        coords_out = getattr(self, '_cached_output_coords', None)
        return resolve_mode(
            self.execution_config, self.max_depth,
            population=population,
            num_inputs=int(coords_in.shape[0]) if coords_in is not None else 3,
            num_outputs=int(coords_out.shape[0]) if coords_out is not None else 1,
            device_bytes=device_bytes,
        )

    def generation_execution_mode(self) -> str:
        """The execution mode a generation of `run_generation` runs: 'depth_offload' when the
        resolution (memory-aware on a GPU) says so, otherwise 'streaming' or 'standard' by the
        build step the generation loop selects (`enable_streaming`). `run_until_threshold`
        reports its route's own mode in its result.

        Returns:
            'standard', 'streaming' or 'depth_offload'.
        """
        if self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD:
            return 'depth_offload'
        return 'streaming' if self.enable_streaming else 'standard'

    def should_use_offloading(self) -> bool:
        """Whether this generation will take the depth-offload path."""
        return self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD

    def _calculate_optimal_pos_chunk_size(
        self,
        pop_size: int,
        num_sources: int,
        target_gb: Optional[float] = None,
        max_nodes: Optional[int] = None,
    ) -> int:
        """Positions per CPPN query chunk. See `execution.chunking`.

        `max_nodes` sizes the dominant forward-carry term; when the caller has
        no transformed-CPPN tuple in hand, the genome default is used.
        """
        from emr_hyperneat.emrhyperneat_base import (
            DEFAULT_MAX_NODES,
        )
        return optimal_position_chunk_size(
            pop_size=pop_size,
            num_sources=num_sources,
            target_gb=self._target_shard_gb if target_gb is None else target_gb,
            max_chunk=self._cppn_pos_chunk_size,
            max_nodes=DEFAULT_MAX_NODES if max_nodes is None else max_nodes,
        )

    def _calculate_population_chunk_size(
        self,
        pop_size: int,
        total_positions: int,
        num_inputs: int = 3,
        num_outputs: int = 1,
    ) -> int:
        """Genomes per chunk that fit the host RAM budget. See `execution.chunking`."""
        return population_chunk_size(
            pop_size=pop_size,
            total_positions=total_positions,
            target_cpu_gb=self._target_cpu_gb,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_population_chunk=self._max_population_chunk,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Memory counters accumulated by the depth-offload path."""
        return {
            'cpu_memory_peak_gb': self._cpu_memory_peak_gb,
            'gpu_transfers': self._gpu_transfers_count,
            'depth_offload_enabled': self.should_use_offloading(),
            'depth_threshold': self._depth_offload_threshold,
            'execution_mode': self.resolve_execution_mode().value,
        }

    def _refuse_node_list_runtime(self, device_id: int) -> None:
        """What changed since create_config that the node-list model cannot apply.

        create_config refuses these in the configuration; imperative setters (configure_sharding,
        set_palette_strategy, an execution mode or a strategy assigned by hand) and the
        multi-device routes' device_id can bring them in later.

        Raises:
            ValueError: A device other than the default, a multi-device strategy, an execution
                mode other than standard, or a palette strategy.
        """
        problems = []
        if device_id != 0:
            problems.append(f"device_id {device_id} (the node list runs on the default device)")
        if self.strategy not in (MultiGPUStrategy.SINGLE_GPU,
                                 MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP):
            problems.append(f"strategy {self.strategy.name}")
        if (self.enable_streaming
                or self.execution_config.mode not in (ExecutionMode.AUTO, ExecutionMode.STANDARD)
                or self.execution_config.resolve(self.max_depth) is not ExecutionMode.STANDARD):
            problems.append("an execution mode other than standard")
        if self._palette_strategy is not None or self._live_strategy is not None:
            problems.append("a palette strategy (the node list has no per-node functions)")
        if self.execution_config.generation_loop != 'python':
            problems.append(
                f"execution.generation_loop {self.execution_config.generation_loop!r} (the node "
                f"list runs the Python loop)"
            )
        if problems:
            raise ValueError("recurrence.network_model 'node_list' cannot apply "
                             + "; ".join(problems))

    def _run_generation_node_list(
        self,
        state: Any,
        cppn_population: Tuple,
        cppns_transformed: Tuple,
        step_times: Dict[str, Any],
        total_start: float,
        device_id: int = 0,
    ) -> Tuple[Any, AlgorithmMetrics]:
        """One generation of HMR's node-list model.

        run_generation_verbose calls it after the ask, the transform and the optional pre-ask
        split. The population's networks come from one compiled call (node_list.NodeListModel),
        are scored by _node_list_forward with the configured fitness mode, and the tell, the
        champion record and the metrics follow as on the grid path. Nothing of the grid is built.
        HMR looped over the genomes in Python, for discovery and again for evaluation.

        Args:
            state: Algorithm state.
            cppn_population: The asked population (nodes, conns).
            cppns_transformed: Its transformed CPPNs.
            step_times: The generation's step timings so far.
            total_start: When the generation started (time.perf_counter).
            device_id: Must be 0.

        Returns:
            Tuple of (new_state, metrics).

        Raises:
            RuntimeError: Before create_config.
            ValueError: A setting changed since create_config that the node list cannot apply
                (_refuse_node_list_runtime).
        """
        settings = self._node_list_settings
        if settings is None:
            raise RuntimeError("the node-list model's settings come from create_config: call it "
                               "first")
        self._refuse_node_list_runtime(device_id)
        pop_size = int(cppn_population[0].shape[0])

        # Discovery: the whole population in one compiled call
        start = time.perf_counter()
        model = node_list.node_list_model(self._jitted_cppn_forward, settings,
                                          self.substrate_input_coords,
                                          self.substrate_output_coords, pop_size)
        networks = model.discover(state, cppns_transformed)
        step_times['step3_phase2_discovery'] = time.perf_counter() - start
        # The grid's arrays describe no node list. Cleared, so a reader of the grid model (the
        # weight serializer, the multi-task evaluation) finds nothing rather than another
        # configuration's arrays.
        for name in ('_cached_W1', '_cached_W2', '_cached_masks_A', '_cached_sparse_hh',
                     '_cached_self_conn', '_cached_act_indices', '_cached_hh_agg_indices',
                     '_cached_ih_agg_indices', '_cached_eval_W1', '_cached_eval_W2',
                     '_cached_eval_sparse_hh', '_cached_active_indices'):
            setattr(self, name, None)
        self._cached_node_list = networks

        # Evaluation, as on the grid path
        start = time.perf_counter()
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        forward_one = _node_list_forward(settings.activate_time)
        genome_args = (networks.W_in, networks.W_hh, networks.W_out)
        fitness_mode_val = self.fitness_mode

        def eval_one(*args):
            outputs = forward_one(inputs_batch, *args)
            return _eval_single_fitness(outputs, targets_batch, fitness_mode_val)

        num_leaves = int(networks.hidden.shape[1])
        eval_chunk_size = (getattr(self.extended_config, 'eval_chunk_size', None)
                           if self.extended_config else None)
        if eval_chunk_size == -1:
            eval_chunk_size = _auto_eval_chunk_size(num_leaves, pop_size)
        if eval_chunk_size and eval_chunk_size > 0 and pop_size > eval_chunk_size:
            fitnesses = chunked_forward_eval(forward_one, fitness_mode_val, inputs_batch,
                                             targets_batch, eval_chunk_size, *genome_args)
        else:
            fitnesses = jax.vmap(eval_one)(*genome_args)
        step_times['step5_evaluation'] = time.perf_counter() - start
        if self._fitness_rescale_variance is not None:
            mse_values = 1.0 - fitnesses
            fitnesses = jnp.maximum(0.0, 1.0 - mse_values / self._fitness_rescale_variance)
        fitnesses, nan_fitness_count = sanitize_fitness(fitnesses)
        self._cached_fitnesses = fitnesses

        # NEAT evolution
        start = time.perf_counter()
        new_state = self._compiled_tell(state, fitnesses)
        step_times['step6_neat_evolution'] = time.perf_counter() - start
        self._record_evaluation(
            forward_one, genome_args, fitnesses, cppn_population, networks.hidden,
            networks.hidden, None, None, None, new_state,
            full_weights={'W1_full': networks.W_in, 'W2_full': networks.W_out,
                          'W_hh_full': networks.W_hh},
        )
        self._current_generation += 1
        self._last_fitnesses = fitnesses

        # Metrics
        total_time = time.perf_counter() - total_start
        best_fitness, mean_fitness, min_fitness, std_fitness = fitness_statistics(
            fitnesses, nan_fitness_count)
        counts = np.asarray(networks.counts)
        column = {name: counts[:, i] for i, name in enumerate(node_list.COUNT_NAMES)}
        hidden_nodes = column['hidden_nodes']
        kept = (column['input_connections'] + column['hidden_connections']
                + column['output_connections'])
        utilization = float(hidden_nodes.mean()) / num_leaves
        custom_metrics = {
            'total_positions': num_leaves,
            'step_timings': step_times,
            'method': 'node_list',
            'network_model': 'node_list',
            'recurrence_mode': self.get_recurrence_mode(),
            'iteration_level': settings.rounds,
            'activate_time': settings.activate_time,
            'cache_hit': False,
            'nan_fitness_count': nan_fitness_count,
            'avg_hidden_nodes': float(hidden_nodes.mean()),
            'min_hidden_nodes': int(hidden_nodes.min()),
            'max_hidden_nodes': int(hidden_nodes.max()),
            'position_utilization': utilization,
            'node_list_connections_mean': float(kept.mean()),
            'node_list_connections_max': int(kept.max()),
            # Genomes HMR would have scored on its first 500 connections in Python set order
            'node_list_over_rn_cap': int(np.sum(kept > node_list.RN_CONNECTION_CAP)),
            'node_list_cleaning_iterations_max': int(column['cleaning_iterations'].max()),
            **{f'node_list_{name}_mean': float(column[name].mean())
               for name in ('input_connections', 'hidden_connections', 'output_connections',
                            'discovered_nodes')},
        }
        num_species, species_sizes, species_fitness = species_summary(state.species, fitnesses)
        topology = topology_summary(networks.W_in, networks.W_out, networks.hidden,
                                    jnp.asarray(column['hidden_connections']))
        hidden_connections = int(column['hidden_connections'].sum())
        self._extended_metrics = EMRRecurrenceMetrics(
            total_connections=int(kept.sum()),
            hidden_to_hidden_connections=hidden_connections,
            direct_hh_connections=hidden_connections,
            fitness=best_fitness,
            activate_time_used=settings.activate_time,
            time_cppn_ask_ms=step_times['step0_cppn_ask_transform'] * 1000,
            time_hh_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            time_evaluation_ms=step_times['step5_evaluation'] * 1000,
            time_neat_evolution_ms=step_times['step6_neat_evolution'] * 1000,
            phase2_discovery_ms=step_times['step3_phase2_discovery'] * 1000,
            forward_pass_ms=step_times['step5_evaluation'] * 1000,
            effective_iteration_level=settings.rounds,
            active_positions_mean=float(hidden_nodes.mean()),
            active_positions_min=int(hidden_nodes.min()),
            active_positions_max=int(hidden_nodes.max()),
            total_positions=num_leaves,
            position_utilization=utilization,
            band_threshold_used=settings.band_threshold,
            max_weight_used=settings.max_weight,
            min_fitness=min_fitness,
            max_fitness=best_fitness,
            std_fitness=std_fitness,
        )
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
            evaluations=pop_size,
            time_elapsed=total_time,
            **topology,
        )
        metrics.custom_metrics = custom_metrics
        return new_state, metrics

    def _build_matrices_streaming(
        self,
        state: Any,
        cppns_transformed: Tuple,
        h_grid: 'HierarchicalGridStructure',
        step_times: Dict[str, float],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build W1, W2, masks_A level-by-level for memory efficiency.

        This method queries the CPPN one hierarchical level at a time instead of all
        positions at once: one level's queries, and the next level's input queries for its
        variance, are resident at once; W1 and W2 are still assembled at full size.

        Args:
            state: Algorithm state
            cppns_transformed: Transformed CPPN parameters
            h_grid: Pre-computed hierarchical grid structure
            step_times: Dict to record timing info

        Returns:
            Tuple of (W1, W2, masks_A):
                W1: (pop, num_inputs, total_positions) - input->hidden weights
                W2: (pop, total_positions, num_outputs) - hidden->output weights
                masks_A: (pop, total_positions) - active position masks
        """
        t_stream_start = time.perf_counter()

        pop_size = cppns_transformed[0].shape[0]
        num_levels = h_grid.num_levels
        total_positions = h_grid.total_positions

        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords
        num_inputs = input_coords.shape[0]
        num_outputs = output_coords.shape[0]

        max_weight = self.max_weight
        weight_thresh = self.weight_threshold
        if self.population_chunk_size > 0:
            # Manual override.
            pop_chunk_size = self.population_chunk_size
        elif self.population_chunk_size == -1:
            # Auto: chunk only when the query's forward carry would not fit the
            # shard budget. Every configuration whose full-population carry fits
            # keeps its measured behavior byte-identical; the gate exists for the
            # depth-7 class of failure, where the unchunked carry alone was
            # a recorded 6.56 GB. Per-genome query
            # results are independent, so chunking never changes values.
            # Tuple order is (nodes, conns, node_mask, conn_mask); the padded node
            # axis is index 0. (Index 1 is max_connections, which equals 2*max_nodes
            # at the defaults, a silent trap for anyone touching this.)
            cppn_max_nodes = cppns_transformed[0].shape[1]
            largest_level = max(
                int(h_grid.level_offsets_static[lv + 1] - h_grid.level_offsets_static[lv])
                for lv in range(num_levels)
            )
            pop_chunk_size = population_chunk_for_query(
                pop_size, largest_level, cppn_max_nodes, self._target_shard_gb)
        else:
            # 0 = explicit no-chunking.
            pop_chunk_size = pop_size

        # LEO configuration
        use_leo = getattr(self, 'use_leo', False)
        leo_threshold = getattr(self, 'leo_threshold', 0.2)
        leo_mode = getattr(self, 'leo_mode', 'compound')

        # A CPPN emits several outputs for two independent reasons: LEO, and
        # dynamic_functions.mode='cppn_output' (which appends a per-node activation
        # index). Both require the multi-output query path, so the count is read here
        # and threaded through alongside use_leo.
        num_cppn_outputs = getattr(self, 'num_cppn_outputs', 1)

        # Pre-allocate weight matrix accumulators
        W1_accumulated = jnp.zeros((pop_size, num_inputs, total_positions), dtype=jnp.float32)
        W2_accumulated = jnp.zeros((pop_size, num_outputs, total_positions), dtype=jnp.float32)

        # Track level variances and reached masks for subdivision
        level_variances = []
        level_reached = []
        level_reached.append(jnp.ones((pop_size, 4), dtype=bool))  # Level 0 always reached

        # Cache for next-level weights (for variance lookahead)
        cached_next_input_weights = None
        # variance_rule legacy: level 0's own variance, set at level 0 and read where level 1's
        # reach is derived
        legacy_v0 = None

        if self.verbose:
            print(f"[Streaming] Processing {num_levels} levels, {total_positions} total positions")
            if use_leo:
                print(f"[Streaming] LEO enabled (threshold={leo_threshold})")

        for level in range(num_levels):
            level_start = h_grid.level_offsets_static[level]
            level_end = h_grid.level_offsets_static[level + 1]
            num_level_positions = level_end - level_start
            level_positions = h_grid.level_positions[level]

            t_level = time.perf_counter()

            # Get reached mask for this level first (before querying)
            if level == 0:
                current_reached_mask = level_reached[0]
            else:
                # Compute reached from previous level's subdivision
                prev_reached = level_reached[level - 1]
                prev_variance = (legacy_v0 if (level == 1 and legacy_v0 is not None)
                                 else level_variances[level - 1])
                # Canonical ES-HyperNEAT subdivides on two clauses:
                #   (level < initial_depth) or (variance > threshold)
                # A parent at level-1 is forced when level <= initial_depth.
                if level <= self.initial_depth:
                    prev_subdivided = prev_reached
                else:
                    prev_subdivided = prev_reached & (prev_variance > self.variance_threshold)

                parent_indices = h_grid.parent_indices[level]
                current_reached_mask = prev_subdivided[:, parent_indices]
                level_reached.append(current_reached_mask)

            # Query CPPN for this level's positions

            # Input weights for this level
            if cached_next_input_weights is not None and level > 0:
                level_input_weights = cached_next_input_weights
            else:
                level_input_weights = query_level_positions(
                    state, cppns_transformed, input_coords, level_positions,
                    True, self._jitted_cppn_forward, pop_chunk_size,
                    device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                    use_leo=use_leo,
                    num_cppn_outputs=num_cppn_outputs,
                )

            if level == 0 and self._legacy_level0_variance:
                # Channel 0 of the first input, the standard path's variance source.
                legacy_v0 = legacy_level0_variance(
                    level_input_weights[:, 0, :, 0] if (use_leo or num_cppn_outputs > 1)
                    else level_input_weights[:, 0, :])

            # Output weights for this level (always query fresh)
            level_output_weights = query_level_positions(
                state, cppns_transformed, output_coords, level_positions,
                False, self._jitted_cppn_forward, pop_chunk_size,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
                use_leo=use_leo,
                num_cppn_outputs=num_cppn_outputs,
            )

            # Compute variance for this level (lookahead to next level)

            if level < num_levels - 1:
                # Query next level for variance computation
                next_level_positions = h_grid.level_positions[level + 1]
                cached_next_input_weights = query_level_positions(
                    state, cppns_transformed, input_coords, next_level_positions,
                    True, self._jitted_cppn_forward, pop_chunk_size,
                    device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                    use_leo=use_leo,
                    num_cppn_outputs=num_cppn_outputs,
                )

                # Compute variance from children
                # Extract weight channel if LEO is enabled
                if use_leo or num_cppn_outputs > 1:
                    # Multi-output: (pop, num_sources, num_positions, num_outputs).
                    # Channel 0 is always the weight; the rest are LEO expression or
                    # the per-node activation index.
                    variance_weights = cached_next_input_weights[:, 0, :, 0]
                else:
                    # Single output: (pop, num_sources, num_positions)
                    variance_weights = cached_next_input_weights[:, 0, :]
                level_variance = compute_single_level_variance(
                    variance_weights,
                    h_grid,
                    level,
                )
                level_variances.append(level_variance)
            else:
                # Finest level - no children, no variance
                level_variances.append(jnp.zeros((pop_size, num_level_positions)))
                cached_next_input_weights = None

            # Apply weight processing and store in accumulators

            active_mask_broadcast = current_reached_mask[:, None, :]

            if use_leo:
                # The standard path's LEO arithmetic, one level at a time: the locality
                # penalty on tanh(w), then leo_sparsify, so a LEO run's network does not depend
                # on the execution mode.
                # tanh-normalized, before gating (the cleanup below deletes W*_raw)
                W1_raw = jnp.tanh(level_input_weights[:, :, :, 0])
                if self.locality_radius is not None:
                    W1_raw = apply_locality_penalty_to_weights(
                        W1_raw, input_coords, level_positions, self.locality_radius
                    )
                W1_level = jnp.where(
                    active_mask_broadcast,
                    leo_sparsify(W1_raw, jnp.tanh(level_input_weights[:, :, :, LEO_CHANNEL]),
                                 leo_mode, weight_thresh, leo_threshold, max_weight),
                    0.0,
                )

                W2_raw = jnp.tanh(level_output_weights[:, :, :, 0])
                if self.locality_radius is not None:
                    # (pop, num_outputs, level_positions) -> positions first for the penalty
                    W2_raw = apply_locality_penalty_to_weights(
                        W2_raw.transpose(0, 2, 1), level_positions, output_coords,
                        self.locality_radius,
                    ).transpose(0, 2, 1)
                W2_level = jnp.where(
                    active_mask_broadcast,
                    leo_sparsify(W2_raw, jnp.tanh(level_output_weights[:, :, :, LEO_CHANNEL]),
                                 leo_mode, weight_thresh, leo_threshold, max_weight),
                    0.0,
                )
            else:
                # Standard mode: Simple weight thresholding
                # level_input_weights shape: (pop, num_sources, num_positions), or
                # (pop, num_sources, num_positions, num_outputs) when the CPPN is
                # multi-output for a reason other than LEO: notably
                # dynamic_functions.mode='cppn_output', which appends a per-node
                # activation index. Only channel 0 is a weight here; the activation
                # channel is consumed by the per-node function selection later in the
                # pipeline, not by matrix construction.
                if level_input_weights.ndim == 4:
                    level_input_weights = level_input_weights[:, :, :, 0]
                    level_output_weights = level_output_weights[:, :, :, 0]

                W1_raw = jnp.tanh(level_input_weights) * max_weight

                # Apply locality penalty for geometry seeding (if configured)
                if self.locality_radius is not None:
                    W1_raw = apply_locality_penalty_to_weights(
                        W1_raw, input_coords, level_positions, self.locality_radius
                    )

                W1_level = jnp.where(
                    active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                    W1_raw,
                    0.0
                )

                # Process output weights
                W2_raw = jnp.tanh(level_output_weights) * max_weight

                # Apply locality penalty for W2: level_positions -> output_coords
                if self.locality_radius is not None:
                    # W2_raw is (pop, num_outputs, level_positions), need to transpose for penalty
                    # (pop, level_positions, num_outputs)
                    W2_for_penalty = W2_raw.transpose(0, 2, 1)
                    W2_for_penalty = apply_locality_penalty_to_weights(
                        W2_for_penalty, level_positions, output_coords, self.locality_radius
                    )
                    # Back to (pop, num_outputs, level_positions)
                    W2_raw = W2_for_penalty.transpose(0, 2, 1)

                W2_level = jnp.where(
                    active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                    W2_raw,
                    0.0
                )

            # Store in accumulators (slice update)
            # Donated on GPU: in-place slice write, no transient second W1/W2
            # (the level-7 786 MB doubling operand).
            W1_accumulated = _accumulate_level_slice(
                W1_accumulated, W1_level, level_start, level_end)
            W2_accumulated = _accumulate_level_slice(
                W2_accumulated, W2_level, level_start, level_end)

            # Explicit memory cleanup
            del level_input_weights, level_output_weights, W1_raw, W2_raw
            del W1_level, W2_level

            if self._streaming_verbose:
                level_time = time.perf_counter() - t_level
                print(f"  Level {level}: {num_level_positions} positions, {level_time*1000:.1f}ms")

        # Build full masks_A from level_reached
        masks_A = jnp.concatenate(level_reached, axis=1)
        # Observable for substrate depth, mirroring the offload path. Pure read.
        self._last_reached_per_level = [
            int(np.count_nonzero(np.asarray(mask))) for mask in level_reached
        ]

        # Final weight matrices
        W1 = W1_accumulated
        W2 = W2_accumulated.transpose(0, 2, 1)  # (pop, num_outputs, total_positions) -> (pop, total_positions, num_outputs)

        step_times['streaming_build'] = time.perf_counter() - t_stream_start

        if self.verbose:
            print(f"[Streaming] Complete: {step_times['streaming_build']*1000:.1f}ms total")

        return W1, W2, masks_A
