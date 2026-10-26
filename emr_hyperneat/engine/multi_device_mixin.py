"""The multi-device routes that run their own pipeline: the process pool, the pmap route and the
FULL_PIPELINE data-parallel route."""
import multiprocessing as mp
import time
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from emr_hyperneat.emrhyperneat_base import (
    batch_query_population_multi_source_chunked,
    compute_hierarchical_variances_batch,
    compute_subdivision_masks_batch,
    get_hierarchical_grid,
)
from emr_hyperneat.engine.discovery import discover_sparse_hh_vectorized_multi_hop
from emr_hyperneat.engine.metrics import EMRRecurrenceMetrics, sanitize_fitness
from emr_hyperneat.engine.pmap import (
    _PMAP_EVAL_DENSE,
    _PMAP_EVAL_HYBRID,
    _full_pipeline_masks_jit,
    _pmap_data_parallel_full_hh,
)


class MultiDeviceMixin:
    """EMRHyperNEAT's multi-device pipelines."""

    def _setup_pmap_evaluation(self) -> None:
        """One-time setup for pmap-based multi-GPU evaluation.

        This method pre-shards the dataset across devices and validates the
        configuration for pmap compatibility. Called once before the first
        pmap generation loop.

        Sets:
            self._pmap_devices: List of JAX devices
            self._pmap_num_devices: Number of devices
            self._pmap_inputs_sharded: Inputs sharded across devices
            self._pmap_targets_sharded: Targets sharded across devices
            self._pmap_samples_per_device: Samples per device (padded if needed)
            self._pmap_original_n_samples: Original sample count (for error aggregation)
        """
        self._pmap_devices = jax.devices()
        self._pmap_num_devices = len(self._pmap_devices)

        if self._pmap_num_devices < 2:
            raise ValueError(
                f"EVAL_ONLY_PARALLEL requires 2+ devices, found {self._pmap_num_devices}. "
                f"Use SINGLE_GPU strategy instead."
            )

        # Get dataset info
        inputs = self._cached_inputs  # (n_samples, n_inputs)
        targets = self._cached_targets  # (n_samples, n_outputs)
        n_samples = inputs.shape[0]
        self._pmap_original_n_samples = n_samples

        # Pad dataset if not divisible by device count
        samples_per_device = (n_samples + self._pmap_num_devices - 1) // self._pmap_num_devices
        padded_total = samples_per_device * self._pmap_num_devices
        pad_needed = padded_total - n_samples

        if pad_needed > 0:
            # Pad zero inputs with 0.5 targets. This route evaluates tanh hidden nodes, a sigmoid
            # output and no bias term (_check_feature_combinations refuses the rest), so a zero row
            # settles at 0 and outputs sigmoid(0) = 0.5 for every genome: it adds exactly 0 to the
            # error sum, which is divided by the original count. Zero targets would add 0.25 per
            # output and score every genome too low (as in 0.1.0).
            inputs_padded = jnp.pad(inputs, ((0, pad_needed), (0, 0)), mode='constant')
            targets_padded = jnp.pad(targets, ((0, pad_needed), (0, 0)), mode='constant',
                                     constant_values=0.5)
            if self.verbose:
                print(f"[EVAL_ONLY_PARALLEL] Padded dataset: {n_samples} -> {padded_total} samples "
                      f"({samples_per_device}/device)")
        else:
            inputs_padded = inputs
            targets_padded = targets

        # Reshape to (num_devices, samples_per_device, features)
        self._pmap_inputs_sharded = inputs_padded.reshape(
            self._pmap_num_devices, samples_per_device, inputs.shape[1]
        )
        self._pmap_targets_sharded = targets_padded.reshape(
            self._pmap_num_devices, samples_per_device, targets.shape[1]
        )
        self._pmap_samples_per_device = samples_per_device

        if self.verbose:
            print(f"[EVAL_ONLY_PARALLEL] Dataset sharded: {self._pmap_inputs_sharded.shape}")
            print(f"[EVAL_ONLY_PARALLEL] Devices: {[str(d) for d in self._pmap_devices]}")

    def _hh_class_flags(self) -> Tuple[bool, bool, bool]:
        """(allow_backward, allow_lateral, allow_self_loops), the pmap pipeline's static arguments.

        Every class and no self loops when no recurrence config is set. The discovery outside the
        pmap (`_pmap_pipeline_hh`) applies the classes of the recurrence configuration itself.
        """
        cfg = self.extended_config
        if cfg is None:
            return (True, True, False)
        return (bool(cfg.allow_backward), bool(cfg.allow_lateral), bool(cfg.allow_self_loops))

    def _run_until_threshold_device_parallel_process(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """ProcessPoolExecutor-based multi-GPU implementation.

        Uses one Python process per device, each with its own JAX runtime and JIT cache, which
        avoids the cross-device errors that ThreadPoolExecutor hits. The processes start at the
        first generation, one single-worker pool per device, each with that device's
        CUDA_VISIBLE_DEVICES already set (start_device_workers): the package initializes JAX at
        import, so a worker could not set it itself (in 0.1.0 every worker ran on the first
        GPU). They serve every generation and stop when the run ends.

        Architecture:
        - Main process: Computes W1, W2, h->h on GPU 0, handles NEAT evolution
        - Worker processes: Each evaluates a population shard on its own device, h->h
          connections with the kernel EVAL_ONLY_PARALLEL uses

        Trade-offs:
        - Pro: no cross-device JIT cache errors
        - Con: CPPN queries still sequential (on GPU 0)
        - Con: IPC overhead (data serialization)
        - A worker that fails raises (combine_shard_fitnesses); in 0.1.0 its shard kept a
          fitness of 0, and with h->h connections every worker failed

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        from ..multi_gpu_worker import (EvalShardInput, combine_shard_fitnesses, evaluate_shard_worker,
                                       start_device_workers)

        devices = jax.devices()
        num_devices = len(devices)

        print(f"[ProcessPoolExecutor] Using {num_devices} worker processes in parallel, "
              f"one per device")

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets

        # Convert task data to numpy for worker IPC
        inputs_np = np.array(jax.device_get(inputs_batch))
        targets_np = np.array(jax.device_get(targets_batch))

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Get spawn context for ProcessPoolExecutor
        ctx = mp.get_context('spawn')

        # Main evolution loop
        # One single-worker pool per device, started at the first generation (none for zero
        # generations) and shut down however the loop ends
        pools = None
        all_fitnesses = None
        try:
            while best_so_far < target_fitness and generation < max_generations:
                gen_start = time.perf_counter()

                # The opt-in pre-ask key split (extra_randkey_split, default off), as
                # run_generation_verbose does it, so a run draws the same keys on every route
                if getattr(self, '_repro_extra_randkey_split', False):
                    _, randkey = jax.random.split(state.randkey)
                    state = state.update(randkey=randkey)

                # CPPN queries and h->h discovery on GPU 0 (sequential)
                cppn_population = self._compiled_ask(state)
                pop_size = cppn_population[0].shape[0]
                cppns_transformed = self._compiled_transform_batch(state, cppn_population)

                # CPPN queries on GPU 0
                input_all_weights = batch_query_population_multi_source_chunked(
                    state, cppns_transformed, input_coords, all_positions,
                    True, self._jitted_cppn_forward, device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )
                output_all_weights = batch_query_population_multi_source_chunked(
                    state, cppns_transformed, output_coords, all_positions,
                    False, self._jitted_cppn_forward, device_id=0,
                    geometry_seeding_enabled=self.geometry_seeding_enabled,
                )

                # Variance + masks
                all_weights_for_variance = input_all_weights[:, 0, :]
                level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
                masks_A = compute_subdivision_masks_batch(
                    level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                    initial_depth=self.initial_depth,
                )

                # H->H discovery
                sparse_hh = None
                gen_hh_count = 0
                allow_hh = self.extended_config is not None and self.extended_config.allow_hidden_to_hidden

                if allow_hh:
                    # Check cache
                    use_cache = self._hh_cache is not None
                    is_cache_hit = False

                    if use_cache:
                        if not self._hh_cache.should_refresh(self._current_generation, None):
                            sparse_hh = self._hh_cache.get_cached()
                            is_cache_hit = sparse_hh is not None
                            if is_cache_hit:
                                cache_hit_count += 1

                    if not is_cache_hit:
                        cache_refresh_count += 1
                        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                            state, cppns_transformed, h_grid, masks_A,
                            self.band_threshold, self.max_weight,
                            self.extended_config, self._jitted_cppn_forward,
                            verbose=False, device_id=0,
                            geometry_seeding_enabled=self.geometry_seeding_enabled,
                            use_leo=self.use_leo,
                            leo_threshold=self.leo_threshold, leo_mode=self.leo_mode,
                            num_cppn_outputs=self.num_cppn_outputs,
                        )
                        if use_cache and sparse_hh is not None:
                            self._hh_cache.update_cache(sparse_hh, None, self._current_generation)

                    if sparse_hh is not None:
                        gen_hh_count = int(jnp.sum(sparse_hh.num_valid))

                # Build W1, W2 matrices
                weight_thresh = 0.1
                max_weight_val = self.max_weight
                active_mask_broadcast = masks_A[:, None, :]
                W1_raw = jnp.tanh(input_all_weights) * max_weight_val
                W1 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                    W1_raw, 0.0
                )
                W2_raw = jnp.tanh(output_all_weights) * max_weight_val
                W2 = jnp.where(
                    active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                    W2_raw, 0.0
                )
                W2 = W2.transpose(0, 2, 1)

                cppn_time = time.perf_counter() - gen_start

                # Parallel evaluation using ProcessPoolExecutor
                eval_start = time.perf_counter()

                # Convert to numpy for IPC
                W1_np = np.array(jax.device_get(W1))
                W2_np = np.array(jax.device_get(W2))

                hh_from_np = None
                hh_to_np = None
                hh_weights_np = None
                hh_valid_np = None
                hh_num_valid_np = None

                if sparse_hh is not None:
                    hh_from_np = np.array(jax.device_get(sparse_hh.from_indices))
                    hh_to_np = np.array(jax.device_get(sparse_hh.to_indices))
                    hh_weights_np = np.array(jax.device_get(sparse_hh.weights))
                    hh_valid_np = np.array(jax.device_get(sparse_hh.valid_mask))
                    hh_num_valid_np = np.array(jax.device_get(sparse_hh.num_valid))

                # Create shards
                shard_size = pop_size // num_devices
                shard_inputs = []

                for d in range(num_devices):
                    shard_start = d * shard_size
                    shard_end = pop_size if d == num_devices - 1 else (d + 1) * shard_size
                    actual_shard_size = shard_end - shard_start

                    shard_input = EvalShardInput(
                        shard_start=shard_start,
                        shard_end=shard_end,
                        shard_size=actual_shard_size,
                        W1=W1_np[shard_start:shard_end],
                        W2=W2_np[shard_start:shard_end],
                        inputs=inputs_np,
                        targets=targets_np,
                        hh_from=hh_from_np[shard_start:shard_end] if hh_from_np is not None else None,
                        hh_to=hh_to_np[shard_start:shard_end] if hh_to_np is not None else None,
                        hh_weights=hh_weights_np[shard_start:shard_end] if hh_weights_np is not None else None,
                        hh_valid=hh_valid_np[shard_start:shard_end] if hh_valid_np is not None else None,
                        hh_num_valid=hh_num_valid_np[shard_start:shard_end] if hh_num_valid_np is not None else None,
                        activate_time=self.activate_time,
                    )
                    shard_inputs.append(shard_input)

                # Submit to the workers, shard d to device d's pool
                if pools is None:
                    pools = start_device_workers(num_devices, ctx)
                futures = [pools[d].submit(evaluate_shard_worker, d, shard_inputs[d])
                           for d in range(num_devices)]
                results = [f.result(timeout=300) for f in futures]

                # Combine results; a failed shard raises
                all_fitnesses = combine_shard_fitnesses(results, pop_size)

                eval_time = time.perf_counter() - eval_start

                # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees
                # it, as on the one-device path
                fitnesses, _ = sanitize_fitness(jnp.array(all_fitnesses))
                total_hh_count = max(total_hh_count, gen_hh_count)

                # Update best
                gen_best = float(jnp.max(fitnesses))
                best_so_far = max(best_so_far, gen_best)

                # NEAT evolution
                state = self._compiled_tell(state, fitnesses)
                generation += 1
                self._current_generation = generation

                if collect_history:
                    history.append(gen_best)

                if self.verbose:
                    elapsed = time.perf_counter() - gen_start
                    print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                          f"h->h={gen_hh_count}, cppn={cppn_time:.2f}s, eval={eval_time:.2f}s, "
                          f"total={elapsed:.2f}s")
        finally:
            if pools is not None:
                for pool in pools:
                    pool.shutdown(wait=False, cancel_futures=True)

        # Store extended metrics (none for zero generations, as on the one-device path)
        if all_fitnesses is not None:
            self._extended_metrics = EMRRecurrenceMetrics(
                hidden_to_hidden_connections=total_hh_count,
                total_positions=h_grid.total_positions,
                variance_threshold_used=self.variance_threshold,
                band_threshold_used=self.band_threshold,
                max_weight_used=self.max_weight,
                min_fitness=float(np.min(all_fitnesses)),
                max_fitness=float(np.max(all_fitnesses)),
                std_fitness=float(np.std(all_fitnesses)),
                cache_hit_count=cache_hit_count,
                cache_refresh_count=cache_refresh_count,
            )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            # HMR's keys: converged, solved, final_fitness, sparse_hh_stats
            **self._hmr_result_keys(
                target_fitness, gen_best if generation else best_so_far, best_so_far
            ),
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _run_until_threshold_pmap(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Native jax.pmap multi-GPU evaluation with zero IPC overhead.

        The multi-device route for recurrent runs, at any iteration level. A sample count
        that is not a multiple of the device count is padded with rows that add no error
        (_setup_pmap_evaluation), so the fitness is a single-device run's. Design:
        - GPU 0: CPPN queries, h->h discovery (cache off by default), W1/W2
        - pmap: W1/W2/h->h replicated, data sharded, evaluation in parallel
        - GPU 0: NEAT evolution

        Benefits:
        - Zero IPC overhead (no Python pickle serialization)
        - H->H discovery at any iteration level, on GPU 0 (cache off by default)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        # One-time pmap setup (shard dataset across devices)
        self._setup_pmap_evaluation()

        # Get hierarchical grid info
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Check if h->h is enabled
        has_hh = self.extended_config and self.extended_config.allow_hidden_to_hidden
        activate_time = self.activate_time if has_hh else 1

        # Tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0
        cache_hit_count = 0
        cache_refresh_count = 0

        # Prepare EMR-HyperNEAT arrays (reusable across generations)
        all_positions_arr = jnp.array(all_positions)
        input_coords_arr = jnp.array(input_coords)
        output_coords_arr = jnp.array(output_coords)

        # Padded sample count (includes any padding from setup)
        n_samples_padded = self._pmap_num_devices * self._pmap_samples_per_device

        if self.verbose:
            print("[EVAL_ONLY_PARALLEL] Starting evolution:")
            print(f"  - Devices: {self._pmap_num_devices}")
            print(f"  - Positions: {total_positions}")
            print(
                f"  - H->H: "
                f"{'enabled (cached if hh_cache_enabled)' if has_hh else 'disabled (feedforward)'}")
            print(f"  - Activate time: {activate_time}")
            print(f"  - Samples: {self._pmap_original_n_samples} original, {n_samples_padded} padded")

        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # CPPN queries (GPU 0)
            cppn_start = time.perf_counter()

            # The opt-in pre-ask key split (extra_randkey_split, default off), as
            # run_generation_verbose does it, so a run draws the same keys on every route
            if getattr(self, '_repro_extra_randkey_split', False):
                _, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            # Get CPPN population and transform
            cppn_population = self._compiled_ask(state)
            pop_size = cppn_population[0].shape[0]  # Get population size from nodes array
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # Batch CPPN queries for input->positions (W1)
            # outgoing=True: input coords as source, all positions as target
            input_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, input_coords_arr, all_positions_arr,
                True, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )  # (pop_size, n_inputs, total_positions)

            # Batch CPPN queries for positions->output (W2)
            # outgoing=False: output coords as source (but we want position->output)
            output_all_weights = batch_query_population_multi_source_chunked(
                state, cppns_transformed, output_coords_arr, all_positions_arr,
                False, self._jitted_cppn_forward,
                device_id=0,
                geometry_seeding_enabled=self.geometry_seeding_enabled,
            )  # (pop_size, n_outputs, total_positions)

            cppn_time_ms = (time.perf_counter() - cppn_start) * 1000

            # Variance -> masks (GPU 0)
            # Compute hierarchical variances and subdivision masks
            # Use input[0]->all_positions as variance source (2D: pop_size, total_positions)
            all_weights_for_variance = input_all_weights[:, 0, :]  # (pop_size, total_positions)
            level_variances = compute_hierarchical_variances_batch(all_weights_for_variance, h_grid, legacy_level0_variance=self._legacy_level0_variance)
            masks_A = compute_subdivision_masks_batch(
                level_variances, self.variance_threshold, h_grid, return_all_masks=False,
                initial_depth=self.initial_depth,
            )

            # H->H discovery on GPU 0 (cache off by default)
            hh_start = time.perf_counter()
            hh_from = None
            hh_to = None
            hh_weights = None
            hh_valid = None

            if has_hh:
                # Check cache
                cache_hit = False
                if self._hh_cache is not None:
                    cache_hit = not self._hh_cache.should_refresh(generation, masks_A)

                if cache_hit:
                    # Cache hit - reuse cached h->h connections
                    cached_hh = self._hh_cache.get_cached()
                    if cached_hh is not None:
                        hh_from = cached_hh.from_indices
                        hh_to = cached_hh.to_indices
                        hh_weights = cached_hh.weights
                        hh_valid = cached_hh.valid_mask
                        # Population total, the basis the single-device path reports
                        # (custom_metrics['total_hh_connections']); genome 0 alone
                        # understated it by a factor of the population size.
                        total_hh_count = int(jnp.sum(hh_valid))
                        cache_hit_count += 1
                else:
                    # Cache miss - discover h->h connections
                    # discover_sparse_hh_vectorized_multi_hop does its own CPPN queries internally
                    sparse_hh = discover_sparse_hh_vectorized_multi_hop(
                        state=state,
                        cppns_transformed=cppns_transformed,
                        h_grid=h_grid,
                        masks_A=masks_A,
                        band_threshold=self.band_threshold,
                        max_weight=self.max_weight,
                        config=self.extended_config,
                        cppn_forward=self._jitted_cppn_forward,
                        pop_chunk_size=pop_size,
                        verbose=self.verbose,
                        device_id=0,
                        geometry_seeding_enabled=self.geometry_seeding_enabled,
                        use_leo=self.use_leo,
                        leo_threshold=self.leo_threshold, leo_mode=self.leo_mode,
                        num_cppn_outputs=self.num_cppn_outputs,
                    )

                    hh_from = sparse_hh.from_indices
                    hh_to = sparse_hh.to_indices
                    hh_weights = sparse_hh.weights
                    hh_valid = sparse_hh.valid_mask
                    total_hh_count = int(jnp.sum(sparse_hh.num_valid))  # Population total

                    # Update cache
                    if self._hh_cache is not None:
                        self._hh_cache.update_cache(sparse_hh, masks_A, generation)
                    cache_refresh_count += 1

            hh_time_ms = (time.perf_counter() - hh_start) * 1000

            # Build W1, W2 matrices (GPU 0)
            # Apply masking and build final weight matrices (same as single-GPU)
            weight_thresh = 0.1  # Same as parent
            max_weight = self.max_weight

            # Broadcast mask: (pop_size, 1, total_positions) for weight masking
            active_mask_broadcast = masks_A[:, None, :]

            # Apply tanh + max_weight scaling, then mask
            W1_raw = jnp.tanh(input_all_weights) * max_weight
            W1 = jnp.where(
                active_mask_broadcast & (jnp.abs(W1_raw) > weight_thresh),
                W1_raw,
                0.0
            )  # (pop_size, n_inputs, total_positions)

            W2_raw = jnp.tanh(output_all_weights) * max_weight
            W2_masked = jnp.where(
                active_mask_broadcast & (jnp.abs(W2_raw) > weight_thresh),
                W2_raw,
                0.0
            )
            W2 = W2_masked.transpose(0, 2, 1)  # (pop, n_outputs, total_positions) -> (pop, total_positions, n_outputs)

            # pmap evaluation (true parallel)
            eval_start = time.perf_counter()

            if has_hh and hh_from is not None:
                # Hybrid mode with h->h connections
                # pmap: W1, W2, h->h replicated; inputs, targets sharded
                partial_errors = _PMAP_EVAL_HYBRID(
                    W1, W2,
                    hh_from, hh_to, hh_weights, hh_valid,
                    self._pmap_inputs_sharded,
                    self._pmap_targets_sharded,
                    activate_time,
                    total_positions,
                )  # (num_devices, pop_size)
            else:
                # Dense feedforward mode
                partial_errors = _PMAP_EVAL_DENSE(
                    W1, W2,
                    self._pmap_inputs_sharded,
                    self._pmap_targets_sharded,
                    activate_time,
                )  # (num_devices, pop_size)

            # Aggregate errors across devices
            total_errors = jnp.sum(partial_errors, axis=0)  # (pop_size,)

            # Compute fitness: 1.0 - MSE
            # Divide by the original sample count, not the padded one; the padded rows add 0 to the
            # sum (see _setup_pmap_evaluation)
            n_outputs_per_sample = self._cached_targets.shape[1]
            mse = total_errors / (self._pmap_original_n_samples * n_outputs_per_sample)
            fitnesses = 1.0 - mse
            # NaN (an overflowing settling, a NaN target) becomes -inf before selection sees it,
            # as on the one-device path
            fitnesses, _ = sanitize_fitness(fitnesses)

            eval_time_ms = (time.perf_counter() - eval_start) * 1000

            # NEAT evolution (GPU 0)
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)

            state = self._compiled_tell(state, fitnesses)
            generation += 1
            self._current_generation = generation

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                elapsed = time.perf_counter() - gen_start
                cache_status = "HIT" if (has_hh and cache_hit_count > 0 and (cache_hit_count + cache_refresh_count) > 0 and cache_hit_count == generation) else ("MISS" if has_hh else "N/A")
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"h->h={total_hh_count} ({cache_status}), "
                      f"cppn={cppn_time_ms:.0f}ms, hh={hh_time_ms:.0f}ms, "
                      f"eval={eval_time_ms:.0f}ms, total={elapsed*1000:.0f}ms",
                      flush=True)

        # Metrics of the final population (none for zero generations, as on the one-device path)
        if generation > 0:
            # Store extended metrics
            self._extended_metrics = EMRRecurrenceMetrics(
                hidden_to_hidden_connections=total_hh_count,
                total_positions=total_positions,
                variance_threshold_used=self.variance_threshold,
                band_threshold_used=self.band_threshold,
                max_weight_used=self.max_weight,
                min_fitness=float(jnp.min(fitnesses)),
                max_fitness=float(jnp.max(fitnesses)),
                std_fitness=float(jnp.std(fitnesses)),
                cache_hit_count=cache_hit_count,
                cache_refresh_count=cache_refresh_count,
            )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            # HMR's keys: converged, solved, final_fitness, sparse_hh_stats
            **self._hmr_result_keys(
                target_fitness, gen_best if generation else best_so_far, best_so_far
            ),
            'sparse_hh_count': total_hh_count,
        }
        if collect_history:
            result['history'] = history

        return result

    def _pmap_pipeline_hh(self, state: Any, cppns_transformed: Tuple, h_grid: Any) -> Tuple:
        """The pmap pipeline's h->h connections, discovered outside the pmap.

        The masks of `_full_pipeline_masks` and the bounded discovery SINGLE_GPU and
        EVAL_ONLY_PARALLEL run (`discover_sparse_hh_vectorized_multi_hop`: by default per genome
        over its own active positions, in buckets of a fixed size), on the first device, over the
        population given. In 0.1.0 the pmapped core discovered them itself, querying every
        genome's CPPN at every pair of grid positions: one buffer of about P x N^2 x 401 bytes
        (69.48 GiB at depth 4 and population 100) whatever the number of devices, so the
        recurrent FULL_PIPELINE_PARALLEL route ran out of memory where SINGLE_GPU ran the same
        configuration on one 24 GB GPU.

        Args:
            state: Algorithm state.
            cppns_transformed: The genomes' transformed CPPNs (pop, ...).
            h_grid: The hierarchical grid.

        Returns:
            ``(masks_A, from_indices, to_indices, weights, valid_mask, num_valid)``, one row per
            genome.
        """
        masks_A = _full_pipeline_masks_jit(
            cppns_transformed, h_grid.all_positions, self._cached_input_coords,
            self.variance_threshold, h_grid.level_sizes_static, h_grid.level_offsets_static,
            h_grid.level_grid_sizes_static, h_grid.parent_indices, h_grid.num_levels,
            h_grid.total_positions, state, self._jitted_cppn_forward,
            self.geometry_seeding_enabled, self.initial_depth, self._legacy_level0_variance,
        )
        sparse_hh = discover_sparse_hh_vectorized_multi_hop(
            state, cppns_transformed, h_grid, masks_A,
            self.band_threshold, self.max_weight,
            self.extended_config, self._jitted_cppn_forward,
            verbose=False,
            device_id=0,
            geometry_seeding_enabled=self.geometry_seeding_enabled,
            use_leo=self.use_leo,
            leo_threshold=self.leo_threshold, leo_mode=self.leo_mode,
            num_cppn_outputs=self.num_cppn_outputs,
        )
        return (masks_A, sparse_hh.from_indices, sparse_hh.to_indices, sparse_hh.weights,
                sparse_hh.valid_mask, sparse_hh.num_valid)

    def _run_until_threshold_data_parallel_full(
        self,
        state: Any,
        problem: Any,
        target_fitness: float,
        max_generations: int,
        collect_history: bool = False,
    ) -> Dict[str, Any]:
        """Data-parallel multi-GPU with full pipeline inside pmap.

        This strategy moves the CPPN queries, variance masks, weight building and evaluation
        inside pmap for balanced memory distribution. With h->h connections the masks and the
        connections are found once, outside the pmap, by the bounded discovery SINGLE_GPU runs
        (`_pmap_pipeline_hh`), and passed in replicated.

        Memory distribution: Each GPU processes the full population on its data shard.

        Pipeline per generation:
        1. CPPN ask + transform (CPU/GPU 0)
        2. Inside pmap (each GPU):
           - CPPN queries for input->all and output<-all
           - Variance/subdivision masks (with h->h: from outside the pmap)
           - H->H connections (direct; iteration level 1 only, no multi-hop expansion): from
             outside the pmap
           - Build W1, W2 matrices
           - Evaluate on data shard
        3. Combine the shards' fitnesses over the real samples (their mean when no
           padding was needed)
        4. NEAT evolution (CPU/GPU 0)

        Args:
            state: Algorithm state
            problem: Problem instance
            target_fitness: Stop threshold
            max_generations: Max generations
            collect_history: Collect per-generation history

        Returns:
            Dict with results
        """
        num_devices = len(jax.devices())

        # Get hierarchical grid
        h_grid = get_hierarchical_grid(self.max_depth)
        total_positions = h_grid.total_positions
        all_positions = h_grid.all_positions

        # Extract static parameters from h_grid for pmap (JIT-compatible Python tuples)
        level_sizes = h_grid.level_sizes_static
        level_offsets = h_grid.level_offsets_static
        level_grid_sizes = h_grid.level_grid_sizes_static
        parent_indices = h_grid.parent_indices
        num_levels = h_grid.num_levels

        # Prepare coordinates
        input_coords = self._cached_input_coords
        output_coords = self._cached_output_coords

        # Use cached problem data
        inputs_batch = self._cached_inputs
        targets_batch = self._cached_targets
        n_samples = inputs_batch.shape[0]

        # Pad samples to be divisible by num_devices, with zero inputs and 0.5 targets: the pipeline
        # evaluates tanh hidden nodes, a sigmoid output and no bias term (_refuse_pmap_pipeline_gaps),
        # so a zero row settles at 0, outputs sigmoid(0) = 0.5 for every genome and adds no error.
        # The shards are then combined over the real samples only (below). Copies of the last
        # sample (0.1.0) counted it once per copy.
        remainder = n_samples % num_devices
        if remainder != 0:
            pad_samples = num_devices - remainder
            inputs_padded = jnp.pad(inputs_batch, ((0, pad_samples), (0, 0)), mode='constant')
            targets_padded = jnp.pad(targets_batch, ((0, pad_samples), (0, 0)), mode='constant',
                                     constant_values=0.5)
            padded_n_samples = n_samples + pad_samples
        else:
            pad_samples = 0
            inputs_padded = inputs_batch
            targets_padded = targets_batch
            padded_n_samples = n_samples

        per_gpu_samples = padded_n_samples // num_devices

        # Shard dataset: (n_samples, features) -> (num_gpus, per_gpu_samples, features)
        inputs_sharded = inputs_padded.reshape(num_devices, per_gpu_samples, -1)
        targets_sharded = targets_padded.reshape(num_devices, per_gpu_samples, -1)

        self._refuse_pmap_pipeline_gaps('FULL_PIPELINE_PARALLEL')
        # Extended config parameters
        allow_hh = self.extended_config.allow_hidden_to_hidden if self.extended_config else False
        iteration_level = self._pmap_iteration_level(allow_hh)
        max_sparse_conns = self.extended_config.max_sparse_conns if self.extended_config else 1000

        if self.verbose:
            print(f"[UnifiedExtended] FULL_PIPELINE_PARALLEL: {num_devices} GPUs, "
                  f"positions={total_positions}, samples={n_samples} ({per_gpu_samples}/GPU)")
            print(f"  h->h enabled: {allow_hh}, iteration_level: {iteration_level}")

        # History tracking
        history = [] if collect_history else None
        best_so_far = -float('inf')
        generation = 0
        total_hh_count = 0  # Track h->h connections across generations

        # Track substrate efficiency metrics across generations (for averaging)
        cumulative_active = 0.0
        cumulative_w1_nonzero = 0.0
        cumulative_w2_nonzero = 0.0
        cumulative_w1_total = 0.0
        cumulative_w2_total = 0.0
        generation_count = 0

        # Cache tracking
        cache_hit_count = 0
        cache_refresh_count = 0
        last_hh_count = -1  # -1 so first gen is always a "refresh"

        # Python while loop for control flow
        while best_so_far < target_fitness and generation < max_generations:
            gen_start = time.perf_counter()

            # CPPN ask + transform (on GPU 0), after the opt-in pre-ask key split
            # (extra_randkey_split, default off) as run_generation_verbose does it. In 0.1.0
            # this read the base class's attribute, which defaults on, so these routes drew other
            # keys than SINGLE_GPU from generation 1
            if getattr(self, '_repro_extra_randkey_split', False):
                _, randkey = jax.random.split(state.randkey)
                state = state.update(randkey=randkey)

            cppn_population = self._compiled_ask(state)
            cppns_transformed = self._compiled_transform_batch(state, cppn_population)

            # With h->h, the masks and the connections are found once, outside the pmap, by the
            # bounded discovery SINGLE_GPU runs
            hh = (self._pmap_pipeline_hh(state, cppns_transformed, h_grid)
                  if allow_hh and iteration_level > 0 else None)

            # Full pipeline inside pmap
            # Each GPU runs CPPN queries, variance, weight building, and evaluation on its data
            # shard, with full population replicated
            (partial_fitnesses, hh_counts_sharded,
             active_sums, w1_nz_sums, w2_nz_sums, w1_totals, w2_totals) = _pmap_data_parallel_full_hh(
                cppns_transformed,           # None: replicated
                all_positions,               # None: replicated
                input_coords,                # None: replicated
                output_coords,               # None: replicated
                inputs_sharded,              # 0: sharded
                targets_sharded,             # 0: sharded
                self.variance_threshold,     # None: replicated
                self.band_threshold,         # None: replicated
                self.max_weight,             # None: replicated
                0.1,                         # weight_thresh - None: replicated
                self.activate_time,          # static
                level_sizes,                 # static
                level_offsets,               # static
                level_grid_sizes,            # static
                parent_indices,              # static
                num_levels,                  # static
                total_positions,             # static
                max_sparse_conns,            # static
                iteration_level,             # static
                allow_hh,                    # static
                state,                       # None: replicated
                self._jitted_cppn_forward,   # static
                self.geometry_seeding_enabled,  # static - Risi & Stanley 2012
                self.initial_depth,          # static
                self._legacy_level0_variance,  # static (24)
                *self._hh_class_flags(),  # static (25-27): allow_backward, allow_lateral, allow_self_loops
                hh,                          # None: replicated (28)
            )

            # Aggregate across data shards. Each device scored 1 - MSE over its per_gpu_samples rows.
            # Without padding the shards are equal parts of the dataset, so their mean is the
            # fitness. With padding the padded rows add no error, so the dataset's MSE is
            # per_gpu_samples / n_samples times the sum of the shards' MSEs; a shard at -inf (NaN)
            # keeps the genome at -inf.
            if pad_samples == 0:
                fitnesses = jnp.mean(partial_fitnesses, axis=0)  # (pop_size,)
            else:
                fitnesses = 1.0 - (per_gpu_samples / n_samples) * jnp.sum(
                    1.0 - partial_fitnesses, axis=0)

            # Aggregate h->h counts (same population on each GPU, so take first)
            # Each GPU computes h->h for the full population, they should be identical
            gen_hh_count = int(hh_counts_sharded[0])  # Take from first GPU
            self._last_pmap_hh_count = gen_hh_count  # for records and tests
            total_hh_count = max(total_hh_count, gen_hh_count)  # Track max across generations

            # Aggregate substrate efficiency metrics (same population, take from first GPU)
            cumulative_active += float(active_sums[0])
            cumulative_w1_nonzero += float(w1_nz_sums[0])
            cumulative_w2_nonzero += float(w2_nz_sums[0])
            cumulative_w1_total += float(w1_totals[0])
            cumulative_w2_total += float(w2_totals[0])
            generation_count += 1

            # Track cache stats (only for modes with h->h discovery)
            if gen_hh_count > 0:
                if gen_hh_count == last_hh_count:
                    cache_hit_count += 1
                else:
                    cache_refresh_count += 1
                last_hh_count = gen_hh_count

            # Ensure fitnesses are on primary device and fully materialized
            # This prevents device sharding issues from pmap affecting _compiled_tell
            fitnesses = jax.device_put(fitnesses, jax.devices()[0])
            fitnesses = jax.block_until_ready(fitnesses)

            # NEAT evolution
            state = self._compiled_tell(state, fitnesses)
            self._current_generation += 1

            # Update tracking
            gen_best = float(jnp.max(fitnesses))
            best_so_far = max(best_so_far, gen_best)
            generation += 1

            if collect_history:
                history.append(gen_best)

            if self.verbose:
                gen_time = time.perf_counter() - gen_start
                print(f"  Gen {generation}: best={gen_best:.4f}, overall={best_so_far:.4f}, "
                      f"time={gen_time:.3f}s")

        # Metrics of the final population (none for zero generations, as on the one-device path)
        if generation > 0:
            # Compute average substrate efficiency metrics
            pop_size = cppns_transformed[0].shape[0]
            if generation_count > 0:
                avg_active = cumulative_active / generation_count / pop_size
                position_util = (cumulative_active / generation_count) / total_positions / pop_size
                avg_w1_density = cumulative_w1_nonzero / cumulative_w1_total if cumulative_w1_total > 0 else 0.0
                avg_w2_density = cumulative_w2_nonzero / cumulative_w2_total if cumulative_w2_total > 0 else 0.0
            else:
                avg_active = 0.0
                position_util = 0.0
                avg_w1_density = 0.0
                avg_w2_density = 0.0

            # Store extended metrics including h->h count and substrate efficiency
            self._extended_metrics = EMRRecurrenceMetrics(
                hidden_to_hidden_connections=total_hh_count,
                total_positions=total_positions,
                # Config echo (what was actually used)
                variance_threshold_used=self.variance_threshold,
                band_threshold_used=self.band_threshold,
                max_weight_used=self.max_weight,
                # Population diversity from final fitnesses
                min_fitness=float(jnp.min(fitnesses)),
                max_fitness=float(jnp.max(fitnesses)),
                std_fitness=float(jnp.std(fitnesses)),
                # Substrate efficiency (averaged over generations)
                active_positions_mean=avg_active,
                position_utilization=position_util,
                w1_density=avg_w1_density,
                w2_density=avg_w2_density,
                # Cache stats
                cache_hit_count=cache_hit_count,
                cache_refresh_count=cache_refresh_count,
                # The pmap pipeline discovers at level 1 only: levels above 1 are refused on entry,
                # and level 0 runs as level 1 (`_pmap_iteration_level`).
                effective_iteration_level=1 if allow_hh else 0,
            )

        result = {
            'generations': generation,
            'best_fitness': best_so_far,
            'state': state,
            # HMR's keys: converged, solved, final_fitness, sparse_hh_stats
            **self._hmr_result_keys(
                target_fitness, gen_best if generation else best_so_far, best_so_far
            ),
        }
        if collect_history:
            result['history'] = history

        return result
