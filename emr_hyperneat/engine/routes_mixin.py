"""Running until a threshold: the route dispatch of run_until_threshold, the one-device and split loops,
the compiled generation loop, and the island model."""
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import MultiGPUStrategy, get_hierarchical_grid
from emr_hyperneat.engine.discovery import _sparse_cap_run_start, sparse_cap_run_record
from emr_hyperneat.engine.metrics import float32_threshold
from emr_hyperneat.engine.parallel import (
    DeviceSplit,
    SPLIT_STRATEGIES,
    island_seed,
    make_device_split,
    migration_targets,
    remap_hidden_keys,
)
from emr_hyperneat.execution import ExecutionMode

if TYPE_CHECKING:  # the class this mixin is a base of, for the annotation only
    from emr_hyperneat.emrhyperneat import EMRHyperNEAT


class RoutesMixin:
    """EMRHyperNEAT's routes."""

    # Multi-GPU run_until_threshold() Override

    #: The strategies run_until_threshold runs on more than one device (each on one when only one is visible).
    MULTI_DEVICE_ROUTES = (MultiGPUStrategy.FULL_PIPELINE_PARALLEL, MultiGPUStrategy.EVAL_ONLY_PARALLEL,
                           MultiGPUStrategy.POPULATION_PARALLEL_PROCESS)

    def get_num_devices(self) -> int:
        """Devices the strategy's route in run_until_threshold uses.

        Every visible device for FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL and
        POPULATION_PARALLEL_PROCESS; a split route's grid (`_route_device_split`); the devices that
        host islands under ISLAND_MODEL; one for SINGLE_GPU, SINGLE_GPU_PYTHON_LOOP and STREAMING.
        run_until_threshold runs the multi-device routes on one device when only one is visible, and
        FULL_PIPELINE_PARALLEL also on an exhaustive or small dataset; its result records the
        devices a run used. The base class counted devices by the name.

        Returns:
            The number of devices.
        """
        if self.strategy in SPLIT_STRATEGIES:
            return len(self._route_device_split().device_ids) if len(jax.devices()) >= 2 else 1
        if self.strategy == MultiGPUStrategy.ISLAND_MODEL and self._island_plan is not None:
            return len(set(self._island_placement()))
        return len(jax.devices()) if self.strategy in self.MULTI_DEVICE_ROUTES else 1

    def _route_device_split(self) -> DeviceSplit:
        """A split route's device grid: emr_hyperneat.parallel's counts, or every
        visible device along the route's axis (POPULATION_PMAP the genomes, POSITION_SHARDING the
        grid positions). HYBRID has its counts from the configuration.

        Raises:
            ValueError: HYBRID without emr_hyperneat.parallel, or a grid larger than the visible
                devices.
        """
        population, position = self._parallel_grid or (None, None)
        visible = len(jax.devices())
        if self.strategy == MultiGPUStrategy.POPULATION_PMAP:
            return make_device_split(population or visible, 1)
        if self.strategy == MultiGPUStrategy.POSITION_SHARDING:
            return make_device_split(1, position or visible)
        if population is None or position is None:
            raise ValueError(
                "multi_gpu_strategy HYBRID splits the genomes and the grid positions at once: set "
                "emr_hyperneat.parallel.population_devices and position_devices"
            )
        return make_device_split(population, position)

    @staticmethod
    def _record_route(result: Dict[str, Any], route: str, devices: int) -> Dict[str, Any]:
        """Records the route run_until_threshold took, the devices it used and the execution mode it ran.

        A strategy name alone does not say that FULL_PIPELINE_PARALLEL runs one device on an exhaustive dataset such
        as XOR, so a run could carry a multi-device label it did not run. The one-device route reports its own mode (depth offload replaces the generation even under the
        STREAMING route); the multi-device routes run their own on-device pipeline, 'standard'.

        Args:
            result: The route's result dictionary.
            route: SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL,
                POPULATION_PARALLEL_PROCESS, or a split route (POPULATION_PMAP, POSITION_SHARDING,
                HYBRID).
            devices: The devices the route used.

        Returns:
            The same dictionary, with 'route', 'devices' and 'execution_mode'.
        """
        result['route'] = route
        result['devices'] = devices
        result.setdefault('execution_mode', 'standard')
        sparse_cap = sparse_cap_run_record()
        if sparse_cap is not None:
            # The cap the run ended with, its champion's, and how it got there
            result['sparse_cap'] = sparse_cap
        return result

    def _refuse_ignored_execution_mode(self, route: str, split_route: bool = False) -> None:
        """A multi-device route runs its own on-device pipeline: an explicitly configured streaming or depth
        offload would be ignored, so it is refused. AUTO leaves the choice to the engine, and the route's record says
        'standard'.

        Raises:
            ValueError: execution.mode is streaming or depth_offload.
        """
        if not split_route:
            self._refuse_palette_evolution(route)
        if self.execution_config.generation_loop == 'while_loop':
            # The compiled generation loop is one device's
            raise ValueError(
                f"{route} runs its own pipeline on several devices, and execution.generation_loop "
                "'while_loop' runs one device: use SINGLE_GPU with it, or generation_loop 'python'")
        if self.execution_config.mode in (ExecutionMode.STREAMING, ExecutionMode.DEPTH_OFFLOAD):
            raise ValueError(
                f"{route} runs its own on-device pipeline and would ignore execution.mode "
                f"{self.execution_config.mode.value!r}; use SINGLE_GPU or STREAMING for a memory "
                "mode")

    def run_until_threshold(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
        streaming_chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Run multiple generations with h->h discovery support.

        Routes to appropriate implementation based on strategy:

        Nine strategies

        Decision tree:
          Dataset fits on single GPU?
          |- yes -> SINGLE_GPU
          `- no  -> Multiple GPUs available?
                   |- no  -> none (STREAMING splits the CPPN queries by level, not the data)
                   `- yes -> Recurrent/h->h mode?
                            |- yes -> EVAL_ONLY_PARALLEL (any iteration level)
                            `- no  -> FULL_PIPELINE_PARALLEL (data parallel)

          POPULATION_PARALLEL_PROCESS: not recommended (see MultiGPUStrategy)

        1. SINGLE_GPU (default):
           Single GPU execution using run_generation_verbose()
           Uses all unified extended features

        2. FULL_PIPELINE_PARALLEL:
           Full pipeline runs on each GPU: CPPN -> variance -> W1/W2 -> eval
           Dataset is split across GPUs, pipeline is replicated
           Recommended for: Feedforward mode, large statistical datasets (MNIST)
           Memory: ~balanced across GPUs

        3. STREAMING:
           One device, execution mode streaming: the CPPN is queried one level at a
           time, and the dataset stays on the device
           Recommended for: substrates too large to query at once

        4. EVAL_ONLY_PARALLEL:
           Only evaluation parallelized, CPPN/h->h runs once on GPU 0
           Keeping discovery on one device lets it run multi-hop, and lets the h->h
           cache (off by default) survive across generations
           Recommended for: Recurrent (h->h) modes
           Memory: every GPU holds W1/W2, h->h and its data shard; GPU 0 also holds
           the CPPN queries and the h->h discovery

        5. POPULATION_PARALLEL_PROCESS:
           ProcessPoolExecutor-based parallel processing
           One Python process per device, each started with that device's
           CUDA_VISIBLE_DEVICES and its own JAX runtime
           Not recommended: the least exercised path here, see MultiGPUStrategy

        6. POPULATION_PMAP, POSITION_SHARDING, HYBRID (the split routes):
           SINGLE_GPU's generation with its one-output CPPN query and its evaluation spread
           over a grid of devices (emr_hyperneat.parallel), everything else on the first one
           Every feature of the standard generation runs; on a CPU each generation equals
           SINGLE_GPU's bit for bit

        7. ISLAND_MODEL:
           Islands of the population (emr_hyperneat.islands), each a SINGLE_GPU run on its
           device at its own population and seed, in lockstep, with migration of their best
           genomes every migration_interval generations

        Fallback behavior:
        - Multi-GPU strategies fall back to SINGLE_GPU if only 1 GPU available
        - FULL_PIPELINE_PARALLEL falls back to SINGLE_GPU for exhaustive datasets

        Automatic fallback for exhaustive datasets:
        - If problem.is_exhaustive_dataset is True, FULL_PIPELINE_PARALLEL is disabled
          and single-GPU mode is used instead, regardless of GPU count, as it is for a
          dataset with fewer than 16 samples per device.
        - Statistical datasets (MNIST, ImageNet) can still use FULL_PIPELINE_PARALLEL.

        SINGLE_GPU, STREAMING, the split routes, the islands and every one-device fallback run the
        unified extended features (multi-hop h->h discovery, sparse forward pass, caching); the
        other multi-device routes do not all: FULL_PIPELINE_PARALLEL's data-parallel run refuses
        h->h above iteration level 1, and POPULATION_PARALLEL_PROCESS evaluates the h->h
        connections it discovers on GPU 0 with the kernel EVAL_ONLY_PARALLEL uses.

        execution.generation_loop 'while_loop' (HMR's GPU loop) runs the one-device route as one
        compiled lax.while_loop over the plain feed-forward generation, performance only, up to
        rounding; it refuses every other setting and the multi-device routes refuse it
        (_run_until_threshold_while_loop).

        Args:
            state: Initialized algorithm state
            problem: Problem instance (checked for is_exhaustive_dataset property)
            target_fitness: Stop when best_fitness >= target
            max_generations: Maximum generations
            collect_history: If True, collect per-generation history
            streaming_chunk_size: Chunk size for streaming (unused in extended)

        Returns:
            Dict with 'generations', 'best_fitness', 'state', 'history' (optional), 'route' (the route taken:
            SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL,
            POPULATION_PARALLEL_PROCESS, a split route or ISLAND_MODEL), 'devices' (the devices it
            used) and 'execution_mode' (the mode it ran: standard, streaming or depth_offload);
            ISLAND_MODEL adds 'islands', 'champion_island' and 'migrations'
            (_run_until_threshold_islands)

        Raises:
            ValueError: a multi-device route with an explicit execution.mode streaming or depth_offload, which
                it would ignore; a strategy without a route on two or more devices (assigned to
                ``self.strategy`` by hand); POPULATION_PARALLEL_PROCESS with features its workers drop.
        """
        if self.network_model == 'node_list':
            # The node-list model runs the one-device loop only: a strategy assigned by hand or an
            # execution mode set since create_config raises here, before any route
            self._refuse_node_list_runtime(0)
        _sparse_cap_run_start(None if self.network_model == 'node_list' else self.extended_config)
        num_devices = len(jax.devices())

        # Check if FULL_PIPELINE_PARALLEL sharding is appropriate for this problem
        # FULL_PIPELINE_PARALLEL shards the data across GPUs (each GPU evaluates a subset)
        # and averages the shards' fitnesses; a problem can still ask for one device
        # (is_exhaustive_dataset), and a small dataset always gets one
        n_samples = self._cached_inputs.shape[0] if hasattr(self, '_cached_inputs') and self._cached_inputs is not None else 0

        # Check 1: the problem asks for one device (is_exhaustive_dataset)
        # Examples: XOR (4 samples), logic gates, parity problems
        is_exhaustive = getattr(problem, 'is_exhaustive_dataset', False)

        # Check 2: a dataset with fewer than 16 samples per device runs on one device
        # (a rule of thumb; sharded, the fitness would be the dataset's whatever the shard size)
        MIN_SAMPLES_PER_GPU = 16
        has_enough_samples = n_samples >= (MIN_SAMPLES_PER_GPU * num_devices)

        # Can use data-parallel only if: NOT exhaustive AND enough samples
        can_use_data_parallel = (not is_exhaustive) and has_enough_samples

        # Only run_generation_verbose records a champion for evaluate_on_data; the
        # multi-GPU loops below record none, and an earlier record would be stale
        self._last_eval = None

        # The island model runs on any number of devices: islands of the population evolve apart, in
        # lockstep, each a SINGLE_GPU run of its own population and seed
        if self.strategy == MultiGPUStrategy.ISLAND_MODEL:
            plan = self._island_plan
            if plan is None:
                raise ValueError("multi_gpu_strategy ISLAND_MODEL needs emr_hyperneat.islands, "
                                 "read at create_config")
            placement = self._island_placement()
            print(
                f"[Strategy] ISLAND_MODEL: {plan.count} islands of {list(plan.populations)} "
                f"genomes on devices {list(placement)}, in lockstep, migration every "
                f"{plan.migration_interval} generations (0: none)"
            )
            return self._record_route(self._run_until_threshold_islands(
                problem, target_fitness, max_generations, collect_history
            ), 'ISLAND_MODEL', len(set(placement)))

        # A strategy with no route below would run the one-device loop under its own name, on any
        # number of devices. The constructor and the configuration key map every name through
        # STRATEGY_NAME_MAP and raise for a name without a route, so this guard fires only for a
        # strategy assigned to self.strategy by hand.
        routed = (
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
            MultiGPUStrategy.STREAMING,
        ) + SPLIT_STRATEGIES
        one_device = (MultiGPUStrategy.SINGLE_GPU, MultiGPUStrategy.BASELINE,
                      MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP)
        if num_devices >= 2 and self.strategy not in routed + one_device:
            raise ValueError(
                f"strategy {self.strategy.name} has no route in run_until_threshold, which runs "
                "SINGLE_GPU, STREAMING, FULL_PIPELINE_PARALLEL, EVAL_ONLY_PARALLEL and "
                f"POPULATION_PARALLEL_PROCESS; on {num_devices} devices it would run on one. A "
                "name given to the constructor or to emr_hyperneat.multi_gpu_strategy maps to one "
                "of these (STRATEGY_NAME_MAP), so this strategy was assigned by hand.")

        # Route based on strategy
        # 5 available GPU strategies (legacy aliases normalized at config parsing)
        # 1. SINGLE_GPU          - Single GPU execution (default)
        # 2. FULL_PIPELINE_PARALLEL - Multi-GPU data parallel (OOM prevention)
        # 3. STREAMING           - one device, CPPN queried level by level (memory)
        # 4. EVAL_ONLY_PARALLEL  - Multi-GPU for recurrent/h->h modes
        # 5. POPULATION_PARALLEL_PROCESS - ProcessPool, one process per GPU (not recommended)
        if num_devices >= 2:
            if self.strategy == MultiGPUStrategy.FULL_PIPELINE_PARALLEL:
                if can_use_data_parallel:
                    # Data-parallel: full pipeline inside pmap, balanced memory
                    print(f"[Strategy] FULL_PIPELINE_PARALLEL: Using {num_devices} GPUs")
                    print(f"  - Sharding {n_samples} samples across GPUs ({n_samples // num_devices} samples/GPU)")
                    print("  - Best for: Large statistical datasets (MNIST, ImageNet)")
                    print("  - Each GPU evaluates full population on its data shard, fitnesses "
                          "combined over the real samples")
                    self._refuse_ignored_execution_mode('FULL_PIPELINE_PARALLEL')
                    return self._record_route(self._run_until_threshold_data_parallel_full(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'FULL_PIPELINE_PARALLEL', num_devices)
                else:
                    # The problem asks for one device (is_exhaustive_dataset), or the dataset
                    # has fewer than 16 samples per device: run on one device
                    print("[Strategy] FULL_PIPELINE_PARALLEL requested but falling back to SINGLE_GPU")
                    if is_exhaustive:
                        print(f"  - Reason: the problem sets is_exhaustive_dataset "
                              f"({n_samples} samples), a request for one device, not a fitness "
                              f"requirement")
                        print("  - Problem type: Combinatorial (XOR, logic gates, parity) where every sample is unique")
                        print("  - Why fallback: the flag only; sharded, the fitness would be "
                              "the same: 100% on one device and 0% on the other average 50% "
                              "across both, the dataset's fitness")
                        print("  - Use SINGLE_GPU for such datasets, even with multi-GPU, not POPULATION_PARALLEL_PROCESS")
                    else:
                        print(f"  - Reason: Dataset too small ({n_samples} samples, need {MIN_SAMPLES_PER_GPU * num_devices}+)")
                        print("  - Why fallback: fewer than 16 samples per device; sharded, "
                              "the fitness would be the same")
                        print("  - Use SINGLE_GPU, not POPULATION_PARALLEL_PROCESS, or increase dataset size")
                    return self._record_route(self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'SINGLE_GPU', 1)
            elif self.strategy == MultiGPUStrategy.POPULATION_PARALLEL_PROCESS:
                # The worker processes score tanh nodes and sum aggregation (multi_gpu_worker.py),
                # so a configuration that asks for more raises here rather than running without it.
                dropped = self._features_a_tanh_sum_path_drops() + self._features_offload_drops()
                if dropped:
                    raise ValueError(
                        "POPULATION_PARALLEL_PROCESS evaluates tanh hidden nodes with sum "
                        "aggregation "
                        "and 1-MSE, so it would ignore " + ", ".join(dropped) + ". Use SINGLE_GPU.")
                # ProcessPoolExecutor-based population parallelism: one Python process per
                # device, started with that device's CUDA_VISIBLE_DEVICES and its own JAX runtime
                # (start_device_workers); a worker that fails raises (combine_shard_fitnesses)
                h_grid = get_hierarchical_grid(self.max_depth)
                total_positions = h_grid.total_positions

                print(f"[Strategy] POPULATION_PARALLEL_PROCESS: Using {num_devices} processes "
                      f"with ProcessPoolExecutor")
                print("  - A separate Python process per device (isolated JAX runtime)")
                print("  - Process-parallel execution without JIT cache cross-device errors")
                print(f"  - h->h discovery at depth {self.max_depth} ({total_positions} positions)")
                print(f"  - Splitting population across devices by process, "
                      f"each gets all {n_samples} samples")
                print("  - Trade-off: Higher IPC overhead")

                self._refuse_ignored_execution_mode('POPULATION_PARALLEL_PROCESS')
                return self._record_route(self._run_until_threshold_device_parallel_process(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'POPULATION_PARALLEL_PROCESS', num_devices)

            elif self.strategy == MultiGPUStrategy.EVAL_ONLY_PARALLEL:
                # Native jax.pmap multi-GPU with zero IPC overhead (recommended for h->h)
                h_grid = get_hierarchical_grid(self.max_depth)
                total_positions = h_grid.total_positions
                has_hh = self.extended_config and self.extended_config.allow_hidden_to_hidden

                print("[Strategy] EVAL_ONLY_PARALLEL: Only eval parallelized, h->h cached "
                      "on GPU 0 when hh_cache_enabled (off by default)")
                print("  - CPPN queries + h->h discovery + W1/W2: GPU 0 only "
                      "(h->h cache too, if enabled)")
                print(f"  - Evaluation: jax.pmap across {num_devices} GPUs (true parallel)")
                print(f"  - Data sharding: Each GPU evaluates on {n_samples // num_devices}+ samples")
                print(f"  - H->H caching: "
                      f"{'on GPU 0 if hh_cache_enabled' if has_hh else 'N/A (feedforward)'}")
                print(f"  - Depth {self.max_depth} ({total_positions} positions)")
                print("  - Best for: Recurrent modes at any iteration level "
                      "(h->h caching optional)")

                self._refuse_ignored_execution_mode('EVAL_ONLY_PARALLEL')
                return self._record_route(self._run_until_threshold_pmap(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'EVAL_ONLY_PARALLEL', num_devices)

            elif self.strategy in SPLIT_STRATEGIES:
                # The split routes: the standard generation, its one-output CPPN
                # query and its evaluation spread over a grid of devices, everything else on the
                # first one
                route = self.strategy.name
                split = self._route_device_split()
                self._refuse_ignored_execution_mode(route, split_route=True)
                print(f"[Strategy] {route}: the standard generation, its CPPN query split over "
                      f"{split.population_devices} x {split.position_devices} devices (genomes x "
                      f"grid positions) and its evaluation over all {len(split.device_ids)}")
                return self._record_route(self._run_until_threshold_split(
                    state, problem, target_fitness, max_generations, collect_history, split
                ), route, len(split.device_ids))

            elif self.strategy == MultiGPUStrategy.STREAMING:
                # Streaming: Level-by-level CPPN queries for memory efficiency
                # Only the CPPN queries are split by level; W1 and W2 stay at full size
                h_grid = get_hierarchical_grid(self.max_depth)
                print("[Strategy] STREAMING: Level-by-level CPPN queries for memory efficiency")
                print("  - Purpose: Reduce CPPN-query memory from O(total_positions) to O(max_level_size)")
                print("  - How it works: Queries CPPN one level at a time, accumulates W1/W2")
                print("  - Best for: Depth too large for all CPPN queries to fit in memory")
                print("  - Memory: CPPN queries level by level each generation; W1/W2 full size")
                print(f"  - Depth {self.max_depth} ({h_grid.total_positions} positions)")

                # Enable streaming mode for this run
                original_streaming = self.enable_streaming
                self.enable_streaming = True
                try:
                    return self._record_route(self._run_until_threshold_single_gpu(
                        state, problem, target_fitness, max_generations, collect_history
                    ), 'STREAMING', 1)
                finally:
                    self.enable_streaming = original_streaming

        # Fallback for single GPU or when multi-GPU strategies run on single device
        # Note: Legacy aliases (MULTI_GPU, DATA_PARALLEL, etc.) are mapped to the
        # primary strategies by the constructor and the configuration key alike
        # (STRATEGY_NAME_MAP), so only the primary 5 are checked here
        if self.strategy in (
            MultiGPUStrategy.FULL_PIPELINE_PARALLEL,
            MultiGPUStrategy.EVAL_ONLY_PARALLEL,
            MultiGPUStrategy.POPULATION_PARALLEL_PROCESS,
        ) + SPLIT_STRATEGIES:
            print(f"[Strategy] {self.strategy.name} requested but only {num_devices} GPU available")
            print("  - Falling back to SINGLE_GPU mode")
            print("  - For multi-GPU: Ensure multiple GPUs are visible to JAX")
        elif self.strategy == MultiGPUStrategy.STREAMING:
            # Streaming: Level-by-level CPPN queries for memory efficiency
            h_grid = get_hierarchical_grid(self.max_depth)
            print("[Strategy] STREAMING: Level-by-level CPPN queries for memory efficiency")
            print("  - Purpose: Reduce CPPN-query memory from O(total_positions) "
                  "to O(max_level_size)")
            print("  - Best for: Depth too large for all CPPN queries to fit in memory")
            print("  - Memory: CPPN queries level by level each generation; W1/W2 full size")
            print(f"  - Depth {self.max_depth} ({h_grid.total_positions} positions)")

            # Enable streaming mode for this run
            original_streaming = self.enable_streaming
            self.enable_streaming = True
            try:
                return self._record_route(self._run_until_threshold_single_gpu(
                    state, problem, target_fitness, max_generations, collect_history
                ), 'STREAMING', 1)
            finally:
                self.enable_streaming = original_streaming

        elif self.strategy == MultiGPUStrategy.SINGLE_GPU:
            print("[Strategy] SINGLE_GPU: Using 1 GPU")
            print("  - All computation on primary GPU")
        elif self.strategy not in one_device:
            print(f"[Strategy] {self.strategy.name} requested on {num_devices} device: "
                  f"running the one-device loop")

        # For single GPU, CPU, or fallback: use run_generation_verbose() loop
        return self._record_route(self._run_until_threshold_single_gpu(
            state, problem, target_fitness, max_generations, collect_history
        ), 'SINGLE_GPU', 1)

    def _generation_loop_refusals(self) -> List[str]:
        """What execution.generation_loop 'while_loop' cannot apply: the settings outside the plain
        feed-forward generation, and the host-side steps between generations."""
        refusals = list(self._feedforward_core_blockers())
        if self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD:
            refusals.append("execution.mode 'depth_offload'")
        if getattr(self, 'df_palette_evolution', None) is not None:
            refusals.append("dynamic_functions.palette_evolution (a host step between generations)")
        if self._live_strategy is not None or self._palette_strategy is not None:
            refusals.append("the live palette strategy (a host step between generations)")
        return refusals

    def _clear_generation_records(self) -> None:
        """Drop the records a Python-loop generation keeps: the compiled loop keeps none, and an
        earlier run's would read as this run's (the champion for evaluate_on_data, the cached arrays
        and fitness, the metrics)."""
        self._last_eval = None
        self._extended_metrics = None
        for name in (
            '_cached_cppns_transformed',
            '_cached_W1',
            '_cached_W2',
            '_cached_masks_A',
            '_cached_self_conn',
            '_cached_sparse_hh',
            '_cached_active_indices',
            '_cached_act_indices',
            '_cached_hh_agg_indices',
            '_cached_ih_agg_indices',
            '_cached_eval_W1',
            '_cached_eval_W2',
            '_cached_eval_sparse_hh',
            '_cached_eval_real_positions',
            '_cached_fitnesses',
            '_last_fitnesses',
            '_last_reached_per_level',
            '_cppn_neuromodulation',
        ):
            setattr(self, name, None)

    def _generation_loop_executable(self, max_generations: int) -> Callable:
        """The compiled while_loop for a history capacity, built once and cached until the
        configuration can change.

        Its arguments are the state, the query rows, the data and the threshold (arguments, never
        closure constants, so the executable holds no run's data); the configuration and the
        compiled ask, transform and tell are read when it is traced. It returns (generations, best,
        last generation's best, NaN count, state, each generation's best). The fitness carries are
        in JAX's default float, the fitness's own dtype: float32, or float64 under x64 (float32
        carries would not hold a float64 fitness).
        """
        loop = self._generation_loop_cache.get(max_generations)
        if loop is not None:
            return loop

        def run(state, rows_in, rows_out, inputs, targets, threshold):
            def keep_going(carry):
                generation, best = carry[0], carry[1]
                return (best < threshold) & (generation < max_generations)

            def generation_step(carry):
                generation, best, _, nan_total, current, history = carry
                gen = self._feedforward_generation(current, (rows_in, rows_out), inputs, targets)
                gen_best = jnp.max(gen.fitnesses)
                return (
                    generation + 1,
                    jnp.maximum(best, gen_best),
                    gen_best,
                    nan_total + gen.nan_count,
                    gen.new_state,
                    history.at[generation].set(gen_best),
                )

            # At least one slot: a zero-length history cannot be indexed, even by a step that never
            # runs
            fitness_float = jax.dtypes.canonicalize_dtype(jnp.float64)
            carry = (
                jnp.int32(0),
                jnp.asarray(-jnp.inf, fitness_float),
                jnp.asarray(-jnp.inf, fitness_float),
                jnp.int32(0),
                state,
                jnp.full((max(max_generations, 1),), -jnp.inf, dtype=fitness_float),
            )
            return jax.lax.while_loop(keep_going, generation_step, carry)

        loop = jax.jit(run)
        self._generation_loop_cache[max_generations] = loop
        return loop

    def _run_until_threshold_while_loop(
        self,
        state: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """The one-device route under execution.generation_loop 'while_loop'.

        The HMR engine's GPU loop (release 0.1.0) with its defects fixed: the plain
        feed-forward generation, _feedforward_generation, compiled once inside lax.while_loop and
        cached, so the generations run on the device and their results come back once. Performance
        only, up to rounding: each generation is the Python loop's arithmetic compiled as one
        program, so a fitness can differ by a float32 ulp; the generation count, the stop rule and
        the key are the Python loop's. It refuses what it cannot apply, each setting named (HMR's
        per-node variants evaluated plain tanh networks in their loops whatever they were configured
        with). It keeps no per-generation record: evaluate_on_data raises after it, and the cached
        arrays are cleared.

        Args:
            state: Algorithm state.
            target_fitness: Stop once a generation's best reaches it (the Python loop's float
                comparison).
            max_generations: The most generations to run (the history's capacity: each value
                compiles once).
            collect_history: Return each generation's best.

        Returns:
            run_until_threshold's keys for the one-device route, plus 'nan_fitness_count' (NaN
            fitness values over the run, told as -inf) and 'generation_loop'.

        Raises:
            ValueError: A setting the loop cannot apply.
        """
        refusals = self._generation_loop_refusals()
        if refusals:
            raise ValueError(
                "execution.generation_loop 'while_loop' runs the plain feed-forward generation "
                "(HMR's GPU loop's scope) and cannot "
                "apply: " + "; ".join(refusals) + ". Use generation_loop 'python' (the default), "
                "which runs every feature.")
        if self.verbose:
            import warnings
            warnings.warn(
                "execution.generation_loop 'while_loop' prints nothing per generation: verbose's "
                "diagnostics need generation_loop 'python'",
                UserWarning,
                stacklevel=3,
            )
        # The palettes a runner may have set are checked, as each Python-loop generation does
        self._check_function_palettes()
        self._clear_generation_records()

        max_generations = max(int(max_generations), 0)
        rows_in, rows_out = self._substrate_query_rows()
        loop = self._generation_loop_executable(max_generations)
        # The Python loop's stop rule: a float32 best against the smallest float32 not below the
        # target; under x64 a float64 best against the target itself
        fitness_float = jax.dtypes.canonicalize_dtype(jnp.float64)
        threshold = (jnp.float32(float32_threshold(target_fitness)) if fitness_float == jnp.float32
                     else jnp.asarray(target_fitness, dtype=fitness_float))
        generations, best, last_best, nan_count, state, history = loop(
            state, rows_in, rows_out, self._cached_inputs, self._cached_targets, threshold)
        generations = int(generations)
        best = float(best)
        self._current_generation += generations

        result = {
            'generations': generations,
            'best_fitness': best,
            'state': state,
            # HMR's keys: converged, solved, final_fitness, sparse_hh_stats (None: no discovery ran)
            **self._hmr_result_keys(
                target_fitness, float(last_best) if generations else best, best
            ),
            'execution_mode': 'standard',
            'generation_loop': 'while_loop',
            'nan_fitness_count': int(nan_count),
        }
        if collect_history:
            result['history'] = [float(value) for value in np.asarray(history)[:generations]]
        return result

    def _before_tell(self, state: Any, fitnesses: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        """What a generation's tell receives: the generation's own state and fitness, or what the
        pre-tell hook makes of them (the island model's migration). The
        generation's records keep its own."""
        hook = self._pre_tell_hook
        return (state, fitnesses) if hook is None else hook(state, fitnesses)

    def _refuse_migration_gaps(self) -> None:
        """What migration between islands cannot carry, refused at create_config
        and again at the run (a palette strategy can be set after create_config).

        Raises:
            ValueError: The h->h cache, which reuses connections by population slot, so an immigrant
                would be scored with the replaced genome's; palette evolution or a palette strategy,
                under which each island's palette would decode an immigrant's per-node functions
                otherwise than its sender's did.
        """
        refused = []
        if self.extended_config is not None and self.extended_config.hh_cache_enabled:
            refused.append(
                "recurrence.hh_cache_enabled (it reuses connections by population slot, so an "
                "immigrant would be scored with the connections of the genome it replaces)"
            )
        if getattr(self, 'df_palette_evolution', None) is not None:
            refused.append(
                "dynamic_functions.palette_evolution (each island's palette would decode an "
                "immigrant's per-node functions otherwise than its sender's did)"
            )
        if self._live_strategy is not None or self._palette_strategy is not None:
            refused.append(
                "a palette strategy (each island's palette would decode an immigrant's per-node "
                "functions otherwise than its sender's did)"
            )
        if refused:
            raise ValueError(
                "emr_hyperneat.islands with migration cannot apply "
                + "; ".join(refused)
                + ": set migration_interval 0, or leave these out"
            )

    def _immigration(self, receiver: 'EMRHyperNEAT',
                     packages: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Callable:
        """The pre-tell hook that delivers a migration to ``receiver``:
        ``packages``, one per sender in island order, each the sender's best genomes (genes and
        fitness) of the generation before. They replace the receiver's worst genomes and fitness
        entries (the higher index on a tie) in what its tell receives, so they compete with the
        fitness they earned. Each package's hidden node keys move above the receiver's largest
        (remap_hidden_keys), one map per package. The receiver's records keep its own evaluation."""
        genome = receiver.neat_algo.genome
        io_keys = np.asarray(list(genome.input_idx) + list(genome.output_idx), dtype=np.float64)

        def hook(state: Any, fitnesses: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
            fitness = np.asarray(fitnesses).copy()
            incoming = sum(len(package[2]) for package in packages)
            # The worst first, the higher index on a tie
            reversed_order = np.argsort(fitness[::-1], kind='stable')
            slots = (len(fitness) - 1 - reversed_order)[:incoming]
            keys = np.asarray(state.pop_nodes[:, :, 0])
            next_key = int(np.nanmax(keys)) + 1
            pop_nodes, pop_conns = (
                np.asarray(state.pop_nodes).copy(),
                np.asarray(state.pop_conns).copy(),
            )
            start = 0
            for nodes, conns, arrived in packages:
                nodes, conns, next_key = remap_hidden_keys(nodes, conns, io_keys, next_key)
                taken = slots[start:start + len(arrived)]
                pop_nodes[taken], pop_conns[taken], fitness[taken] = nodes, conns, arrived
                start += len(arrived)
            told = state.update(pop_nodes=jnp.asarray(pop_nodes, dtype=state.pop_nodes.dtype),
                                pop_conns=jnp.asarray(pop_conns, dtype=state.pop_conns.dtype))
            return told, jnp.asarray(fitness, dtype=fitnesses.dtype)

        return hook

    def _island_placement(self) -> Tuple[int, ...]:
        """The device of each island: emr_hyperneat.islands.devices, or island i on device i mod the
        visible ones."""
        plan = self._island_plan
        visible = len(jax.devices())
        return (
            plan.devices
            if plan.devices is not None
            else tuple(i % visible for i in range(plan.count))
        )

    def _island_engine_config(self, population: int) -> Dict[str, Any]:
        """An island's configuration: the run's, with the island's population and the strategy
        SINGLE_GPU (each island is a one-device run of its own)."""
        config = copy.deepcopy(self._island_source)
        algo_params = (config['algorithm_params']['emrhyperneat']
                       if 'emrhyperneat' in config.get('algorithm_params', {}) else config)
        algo_params['population_size'] = population
        for key in ('pop_size', 'population_size'):
            if key in (algo_params.get('neat') or {}):
                algo_params['neat'][key] = population
        emr = algo_params['emr_hyperneat']
        emr.pop('islands')
        emr['multi_gpu_strategy'] = 'SINGLE_GPU'
        return config

    def _run_until_threshold_islands(
        self,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool,
    ) -> Dict[str, Any]:
        """The island model: emr_hyperneat.islands' islands evolve apart,
        generation by generation in lockstep, each a SINGLE_GPU run of its own.

        Island i is SINGLE_GPU's run at its population and at island_seed(seed, i), seed being
        initialize's: its engine is this configuration with that population and strategy SINGLE_GPU,
        initialized with that seed. The state given to run_until_threshold, the whole population's,
        is not evolved. Each island runs on its device (_island_placement), placed through
        jax.default_device and the generation's device_id, the islands of a lockstep generation in
        index order; placement never changes results. The run stops at the end of the lockstep
        generation in which any island reaches the target.

        Returns:
            run_until_threshold's dictionary: 'generations' (lockstep), 'best_fitness' (over every
            island and generation), and the 'state' of the island whose last generation's champion
            is best (the lowest index on a tie), whose champion this engine records
            (evaluate_on_data, get_best_genome_with_fitness); 'islands', per island its seed,
            population, device, final state and best fitness, and its history with collect_history.

        Raises:
            ValueError: An execution mode other than standard, the compiled generation loop, or no
                initialize (the seeds derive from its seed).
        """
        plan = self._island_plan
        if self._run_seed is None:
            raise ValueError(
                "multi_gpu_strategy ISLAND_MODEL derives its islands' seeds from initialize's: "
                "call initialize first"
            )
        if self.execution_config.generation_loop == 'while_loop':
            raise ValueError("ISLAND_MODEL runs each island's generation in lockstep with the "
                             "others, and execution.generation_loop 'while_loop' compiles one "
                             "run's whole loop: use generation_loop 'python'")
        mode = self.resolve_execution_mode()
        if self.enable_streaming or mode is not ExecutionMode.STANDARD:
            raise ValueError(
                "ISLAND_MODEL runs each island's standard generation, and this configuration runs "
                f"the {'streaming' if self.enable_streaming else mode.value} generation "
                f"(execution.mode {self.execution_config.mode.value!r}): set execution.mode "
                "'standard'")
        visible = jax.devices()
        placement = self._island_placement()
        if max(placement) >= len(visible):
            raise ValueError(
                f"emr_hyperneat.islands.devices {list(placement)} names a device beyond the "
                f"{len(visible)} visible"
            )
        islands = []
        for index, population in enumerate(plan.populations):
            engine = type(self)()
            seed = island_seed(self._run_seed, index)
            with jax.default_device(visible[placement[index]]):
                configured = engine.create_config(self._island_engine_config(population))
                state = engine.initialize(configured, problem, seed=seed)
            islands.append(
                {
                    'island': index,
                    'seed': seed,
                    'population': population,
                    'device': placement[index],
                    'engine': engine,
                    'state': state,
                    'best_fitness': -float('inf'),
                    'history': [],
                }
            )

        if plan.migration_interval > 0:
            self._refuse_migration_gaps()
            for island in islands:
                island['engine']._refuse_migration_gaps()
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        arriving: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        events = []
        while best_so_far < target_fitness and generation < max_generations:
            for island in islands:
                engine = island['engine']
                packages = arriving.get(island['island'])
                engine._pre_tell_hook = (
                    None if packages is None else self._immigration(engine, packages)
                )
                island['evaluated'] = island['state']
                try:
                    with jax.default_device(visible[island['device']]):
                        island['state'], metrics = engine.run_generation_verbose(
                            island['state'], problem, device_id=island['device'])
                finally:
                    engine._pre_tell_hook = None
                island['history'].append(float(metrics.best_fitness))
                island['best_fitness'] = max(island['best_fitness'], island['history'][-1])
            gen_best = max(island['history'][-1] for island in islands)
            best_so_far = max(best_so_far, gen_best)
            generation += 1
            if collect_history:
                history.append(gen_best)
            # Migration after generations M, 2M, ...: each island's best genomes of its own
            # evaluation travel, with the fitness they earned, to their receivers' next tell. One
            # generation of latency, so the order in which the islands of a generation run changes
            # nothing
            arriving = {}
            if plan.migration_interval > 0 and generation % plan.migration_interval == 0:
                targets = migration_targets(plan, self._run_seed, len(events))
                events.append({'generation': generation, 'targets': targets})
                for island, receivers in zip(islands, targets):
                    fitness = np.asarray(island['engine']._last_fitnesses)
                    # The best first, the lower index on a tie
                    top = np.argsort(-fitness, kind='stable')[:plan.migrants]
                    package = (np.asarray(island['evaluated'].pop_nodes)[top],
                               np.asarray(island['evaluated'].pop_conns)[top], fitness[top])
                    for receiver in receivers:
                        arriving.setdefault(receiver, []).append(package)

        # The champion: the best last-generation champion over the islands, the lowest index on a
        # tie among the islands whose returned state holds it first (an immigrant that outscored an
        # island's own champion opens that island's next population, and evaluate_on_data scores the
        # state's first genome)
        def holds_own_champion(island):
            record = island['engine']._last_eval
            if record is None:
                return True
            population = island['engine']._compiled_ask(island['state'])
            nodes, conns = record['champion_genome']
            return (np.array_equal(np.asarray(population[0][0]), nodes, equal_nan=True)
                    and np.array_equal(np.asarray(population[1][0]), conns, equal_nan=True))

        best = max(
            islands,
            key=lambda island: (
                island['history'][-1] if island['history'] else -float('inf'),
                holds_own_champion(island),
                -island['island'],
            ),
        )
        self._last_eval = best['engine']._last_eval
        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': best['state'],
            **self._hmr_result_keys(
                target_fitness, gen_best if generation else best_so_far, best_so_far
            ),
            'execution_mode': 'standard',
            'islands': [
                {
                    key: island[key]
                    for key in ('island', 'seed', 'population', 'device', 'state', 'best_fitness')
                }
                | ({'history': island['history']} if collect_history else {})
                for island in islands
            ],
            'champion_island': best['island'],
            'migrations': events,
        }
        if collect_history:
            result['history'] = history
        return result

    def _run_until_threshold_split(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool,
        split: DeviceSplit,
    ) -> Dict[str, Any]:
        """A split route: the one-device loop with the standard generation's
        one-output CPPN query and its evaluation spread over ``split`` (`_query_population`,
        `_evaluate_population`); everything else runs on the split's first device as it runs on one.
        Every feature of the standard generation runs: hidden-to-hidden discovery at any iteration
        level, per-node functions, aggregation, neuromodulation, LEO, palette evolution. On a CPU
        each generation equals SINGLE_GPU's bit for bit.

        Raises:
            ValueError: An execution mode other than standard (streaming and depth offload have
                their own queries), or a population that leaves a device fewer than two genomes (XLA
                evaluates a batch of one otherwise, by an ulp).
        """
        mode = self.resolve_execution_mode()
        if self.enable_streaming or mode is not ExecutionMode.STANDARD:
            # AUTO can resolve to depth offload here (on a GPU, when the standard generation's
            # estimate exceeds the first device), and the legacy enable_streaming flag sets
            # streaming without execution.mode
            ran = 'streaming' if self.enable_streaming else mode.value
            raise ValueError(
                f"multi_gpu_strategy {self.strategy.name} spreads the standard generation over "
                f"devices, and this configuration runs the {ran} generation (execution.mode "
                f"{self.execution_config.mode.value!r}): set execution.mode 'standard', or use "
                "SINGLE_GPU or STREAMING for a memory mode")
        pop_size = int(state.pop_nodes.shape[0])
        devices = len(split.device_ids)
        if pop_size < 2 * devices:
            raise ValueError(
                f"multi_gpu_strategy {self.strategy.name}: a population of {pop_size} over "
                f"{devices} devices leaves a device fewer than two genomes, which XLA evaluates "
                f"otherwise than a population (a batch of one); use at least {2 * devices} genomes "
                "or fewer devices (emr_hyperneat.parallel)")
        self._device_split = split
        try:
            return self._run_until_threshold_single_gpu(
                state, problem, target_fitness, max_generations, collect_history
            )
        finally:
            self._device_split = None

    def _run_until_threshold_single_gpu(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Single GPU implementation using the existing per-generation code.

        This preserves all unified extended features (h->h discovery, multi-hop,
        caching) by using the existing per-generation implementation.

        Execution mode is honored here, the same way `run_generation` honors it:
        `streaming` through the `enable_streaming` flag that `create_config` resolves,
        and `depth_offload` through the explicit dispatch below. Without that dispatch
        a config asking for `depth_offload` silently ran the standard on-device path,
        which is exactly the out-of-memory failure the mode exists to avoid.

        The dispatch lives here rather than inside `run_generation_verbose` on purpose:
        the offload path does not populate `_cached_W1` / `_cached_W2`, and
        `run_generation_multitask` and the neuromodulation runner read those straight
        after calling that method.

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        if self.execution_config.generation_loop == 'while_loop':
            # HMR's GPU loop: the plain feed-forward generation compiled in
            # lax.while_loop
            return self._run_until_threshold_while_loop(
                state, target_fitness, max_generations, collect_history
            )

        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0

        if self.verbose:
            print(f"[EMR-HyperNEAT] Single GPU mode: max_gens={max_generations}, target={target_fitness}")

        # Resolved once: max_depth and the config do not change inside the loop.
        _use_offload = self.resolve_execution_mode() is ExecutionMode.DEPTH_OFFLOAD
        if _use_offload:
            self._refuse_palette_evolution('depth offload')
            # Imported here rather than at module scope: the strategy modules import from
            # emrhyperneat, which imports this module, so a top-level import would form a cycle.
            from emr_hyperneat.execution.offload import (
                run_generation_offload,
            )
            if self.enable_streaming:
                # Reachable through the MultiGPUStrategy.STREAMING arms above, which
                # force enable_streaming before calling this helper. The two are
                # mutually exclusive memory strategies, and offload wins here for the
                # same reason it wins in `run_generation`: it replaces the whole
                # generation rather than one step inside it. Say so instead of letting
                # the strategy banner claim a path that will not run.
                import warnings
                warnings.warn(
                    "Streaming was requested (MultiGPUStrategy.STREAMING) but the "
                    "execution mode resolves to depth_offload; offload takes "
                    "precedence and streaming will not run. Set one memory strategy, "
                    "not both.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.verbose:
                print("[EMR-HyperNEAT] Execution mode: depth_offload")

        while best_so_far < target_fitness and generation < max_generations:
            if _use_offload:
                state, metrics = run_generation_offload(self, state, problem)
            else:
                state, metrics = self.run_generation_verbose(state, problem)
            gen_best = metrics.best_fitness
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose and generation % 10 == 0:
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}")

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            # HMR's keys: converged, solved, final_fitness, sparse_hh_stats
            **self._hmr_result_keys(
                target_fitness, gen_best if generation else best_so_far, best_so_far
            ),
            # The mode this loop ran (offload replaces the generation, even under the STREAMING route)
            'execution_mode': (
                'depth_offload'
                if _use_offload
                else 'streaming' if self.enable_streaming else 'standard'
            ),
        }
        if collect_history:
            result['history'] = history

        return result
