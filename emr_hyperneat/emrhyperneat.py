"""The public `EMRHyperNEAT` class and its feature layer.

Everything a user configures goes through this class: per-node dynamic activation functions
(the 18-function `ACTIVATION_LIST`, its palettes and their built-in evolution), per-node
aggregation, neuromodulation with multi-task evaluation, LEO (link expression output), h->h
recurrence with its caches, and the continual-learning API (the champion, injection into a
population, the topology freeze, a genome's fitness on another problem).
The substrate engine and multi-GPU dispatch beneath it are in
`emrhyperneat_base.py`; the mechanism end to end is documented in
docs/architecture.md ("How a generation is computed").

Layout
======

The class is assembled here from the mixins of `emr_hyperneat.engine`, and every name of the
engine is imported back, so each keeps this module as its import path (one object per name: a
registry or a cache is the engine's own). A name set on this module, as a test's monkeypatch
does, is also set in the engine modules that hold it (`_EngineFacade`, at the end).

- `engine.config`: key refusal, `EMRConfig` and the recurrence presets, `ForwardPassMode`
- `engine.metrics`: fitness sanitization and the run statistics
- `engine.functions`: activation and aggregation registries, palettes, forward primitives
- `engine.settling`, `engine.neuromodulation`, `engine.multitask`, `engine.leo`: their features
- `engine.substrate_build`: masks, weight matrices, CPPN queries, the streaming build's levels
- `engine.evaluation`: fitness and the forwards; `engine.discovery`: h->h discovery and cache
- `engine.parallel`: device splits, strategy names, islands; `engine.pmap`: the pmap kernels
- the mixins: `generation_mixin` (a generation), `routes_mixin` (run_until_threshold),
  `configuration_mixin`, `modes_mixin`, `multitask_mixin`, `multi_device_mixin`,
  `champion_mixin`, `palette_mixin`, `hmr_names_mixin`

Neuromodulation features (4 levels):
- Level 1: Static Gating - one gate per hidden node, derived from its own input weights, or per
  connection from a dedicated CPPN output (gate_source 'cppn')
- Level 2: Context-Dependent Gating (XdG-style) - Task context modulates gates
- Level 3: Modulatory Neurons - some of a genome's own nodes set a per-case gain for the others
  (activity modulation: Soltoggio's modulatory neurons gate plasticity within a lifetime, and EMR
  has no plasticity)
- Level 4: TRUE Neuromodulation - NT vectors + receptor densities + gain modulation

TRUE neuromodulation (Level 4), key features:
- Neurotransmitter vectors: [DA, 5HT, NE, ACh] (2-6 NT types)
- Receptor density matrices per neuron: (pop_size, total_positions, num_nt_types)
- 7 receptor derivation methods: tanh, abs, normalized, fourier, softmax, orthogonal, phase_shifted
- 3 modulation modes: gating_only, gain_bias_only, full
- H->H connections are also modulated (not just Input->Hidden)
- Multi-task support with 6 aggregation methods (mean, min, weighted, product, softmin, harmonic)

Neuromodulation + dynamic functions (orthogonal features):
- Dynamic functions: per-node structural choice (which activation function)
- Neuromodulation: per-node parametric modulation (how much gain/gating)
- Both can be enabled together

Hybrid sparse-dense architecture
=================================

1. **Dense W1, W2** for input->positions and positions->output (GPU-efficient matmul)
2. **Sparse hidden->hidden** with optional multi-hop expansion via matrix power
3. **Multi-iteration forward** using scatter-add for h->h, matmul for rest

Multi-hop expansion
===================

With A the genome's direct h->h connections (the masked adjacency), iteration_level k forms the
hop terms decay^(h-1) A^h for h = 1..k and keeps, per pair, the term of largest magnitude; each
product is thresholded before the decay. Every hop is an existing direct connection
(`recurrence.multi_hop_second_factor: masked`, the default since 0.2.0). `raw` reproduces
0.1.0's expansion, which multiplied by the raw weights over the population's union of active
positions, so a genome's composite connections could end on positions it did not have active,
ignore its connection class and depend on the other genomes. JIT-compatible because
iteration_level and the second factor are static, so the Python loop compiles away.

Execution modes
===============

1. **Feedforward** (default): Dense W1 -> matmul -> W2 (no hidden->hidden)
2. **Feedforward+H->H** (hybrid): Dense W1/W2 + sparse hidden->hidden, multi-hop at level 2+
3. **Full Recurrent**: All connection types (backward, lateral, self-loop)

Presets
=======

- feedforward: No recurrence (fastest)
- hidden_only: Phase 2 only (forward h->h)
- with_backward: h->h + feedback loops
- with_lateral: h->h + same-layer connections
- with_self: h->h + self-loops
- full_recurrent: Everything (activate_time=20)

The recurrent presets discover direct h->h connections only (iteration_level 1) since 0.2.0; in
0.1.0 they composed two hops (level 2). Set recurrence.iteration_level to compose connections.

Aggregation modes
=================

H->H aggregation implementation
-------------------------------
The `hh_aggregation_mode` config option controls H->H aggregation behavior:

- `hh_aggregation_mode='sum'` (default): Uses fast scatter_add for H->H contributions
- `hh_aggregation_mode='dynamic'`: Enables per-node aggregation for H->H connections
  - Uses segment operations (sum, mean, max, min) instead of scatter_add
  - Each hidden node can use a different aggregation function
  - Aggregation indices computed via same interpretation logic as activation indices

Helper functions for H->H aggregation:
- `segment_sum_2d()`: Batched segment sum using scatter_add
- `segment_max_2d()`: Batched segment max with -inf initialization
- `segment_min_2d()`: Batched segment min with +inf initialization
- `segment_count_2d()`: Count valid elements per segment
- `scatter_aggregate_by_target()`: Main function that applies per-node aggregation

Two-level caching architecture
==============================

This implementation uses two independent caching mechanisms for optimal performance:

Level 1: JAX persistent compilation cache (system-level)
---------------------------------------------------------
Stores compiled XLA functions on disk for reuse across runs.

- Location: /tmp/jax_cache (configurable via JAX_COMPILATION_CACHE_DIR)
- Scope: Reuses compiled functions when array shapes match
- Configuration (set before JAX import):
    os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '1'

Level 2: H->H connection cache (algorithm-level)
------------------------------------------------
HHCacheManager reuses one population-wide h->h connection set for up to hh_refresh_interval
generations. Off by default since 0.2.0 (on in 0.1.0): reproduction moves genomes between
population slots, so between refreshes a genome is scored with the recurrent connections
discovered for whichever genome held its slot. Set hh_cache_enabled: true only to reproduce a
cache-on campaign.

Configuration
=============

```yaml
emr_hyperneat:
  max_depth: 3
  variance_threshold: 0.03

  recurrence:
    preset: hidden_only  # Or configure individually:
    # enabled: true
    # allow_hidden_to_hidden: true
    # allow_backward: false
    # iteration_level: 1  # the presets' level since 0.2.0; 2 composes connections
    # multi_hop_algorithm: matrix_power  # or fori_loop
    # hop_decay_factor: 0.8
    # multi_hop_second_factor: masked  # or raw, the 0.1.0 algebra
    # hh_cache_enabled: false  # default since 0.2.0
    # hh_refresh_interval: 5
```
"""

# ruff: noqa: F401
# The imports are this module's namespace from when the engine was this one file, and every name stays
# importable from here (execution.offload, the tests and outside code import some of them), so they stay
# although the code that used them moved to emr_hyperneat.engine.
import copy
import dataclasses
import functools
import inspect
import numbers
import time
import os
import sys
import types
import numpy as np
import threading
import multiprocessing as mp
from typing import Any, Callable, Dict, Tuple, List, NamedTuple, Optional, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

import jax
import jax.numpy as jnp
from jax import lax

# Import base class and utilities
from emr_hyperneat.emrhyperneat_base import (
    EMRHyperNEATMultiGPU,
    MultiGPUStrategy,
    HierarchicalGridStructure,
    get_hierarchical_grid,
    compute_hierarchical_variances_batch,
    compute_subdivision_masks_batch,
    compute_hierarchical_variances_batch_jit,
    compute_subdivision_masks_batch_jit,
    batch_query_population_multi_source_chunked,
    batch_query_population_positions,
    cppn_query_rows,
    _get_device_specific_query_fn,
    refuse_file_configuration,
    safe_matmul,
)

from emr_hyperneat._compat.core.base_algorithm import AlgorithmMetrics

# Execution modes (memory residency). Only the vocabulary and the chunk arithmetic
# are imported here; the strategy modules import from this one, so the mixins that
# dispatch to them import them at call time to keep the dependency one-way.
from emr_hyperneat.execution import (
    EXECUTION_KEYS,
    ExecutionConfig,
    ExecutionMode,
    parse_execution_config,
)
from emr_hyperneat.execution.chunking import (
    optimal_position_chunk_size,
    population_chunk_for_query,
    population_chunk_size,
)
# The node-list recurrent model, opt-in with recurrence.network_model. The module imports the base
# module only, so this import adds no cycle.
from emr_hyperneat import node_list

# Re-exports: what moved to emr_hyperneat.engine keeps its import path here.
from emr_hyperneat.engine.config import (
    EMRConfig,
    FITNESS_MODES,
    ForwardPassMode,
    HH_CACHE_TRIGGERS,
    NETWORK_MODELS,
    RECURRENCE_PRESETS,
    SETTLING_RULES,
    SPARSE_CAP_POLICIES,
    SparseHiddenConnections,
    _UNKNOWN_KEY_HINTS,
    _explicit_activate_time,
    _refuse_unknown_keys,
    _unknown_keys_message,
    _validated_hh_discovery_mode,
    check_recurrence_switch,
    check_sparse_cap_settings,
    get_recurrence_preset,
    resolve_recurrence,
)
from emr_hyperneat.engine.metrics import (
    EMRRecurrenceMetrics,
    fitness_statistics,
    float32_threshold,
    sanitize_fitness,
    sanitize_fitness_on_device,
    species_summary,
    topology_summary,
)
from emr_hyperneat.engine.leo import (
    LEO_CHANNEL,
    LEO_MODES,
    compute_leo_validity_mask,
    compute_leo_validity_mask_replacement,
    leo_sparsification_hmr_compound,
    leo_sparsify,
    vectorized_weight_sparsification_leo_replacement,
    vectorized_weight_sparsification_with_leo,
)
from emr_hyperneat.engine.multitask import (
    BRANCH_GATING_MODES,
    GENERALIST_BONUS_TYPES,
    MULTITASK_AGGREGATIONS,
    MULTITASK_FITNESS_MODES,
    MultiTaskConfig,
    MultiTaskMetrics,
    ORTHOGONALITY_METRICS,
    TWO_MODULE_MODES,
    _MULTITASK_EVALUATORS,
    _refuse_per_node_in_multitask,
    check_branch_gating_and_penalty,
    compute_fitness,
    compute_modulation_with_branch_gating,
    compute_population_orthogonality,
    refuse_anti_branch_gating,
)
from emr_hyperneat.engine.substrate_build import (
    QUERY_POPULATION_CHUNK,
    WEIGHT_THRESHOLD,
    _accumulate_level_slice,
    _accumulate_level_slice_donated,
    _create_multi_output_query_fn_for_device,
    _get_multi_output_query_fn,
    _per_device_multi_output_cache_lock,
    _per_device_multi_output_vmap_cache,
    apply_locality_penalty_to_weights,
    batch_query_population_multi_source_multi_output,
    batch_query_population_positions_multi_output,
    batch_query_population_self_connections,
    compute_locality_penalty,
    compute_single_level_variance,
    compute_single_level_variance_chunked,
    legacy_level0_variance,
    query_cppn_multi_output,
    query_level_positions,
    query_population_on_device,
    self_connection_rows,
    strong_default_float,
    substrate_masks,
    substrate_query_rows,
    substrate_weight_matrices,
)
from emr_hyperneat.engine.functions import (
    ACTIVATION_FUNCTIONS,
    ACTIVATION_INTERPRETATIONS,
    ACTIVATION_LIST,
    AGGREGATION_FUNCTIONS,
    AGGREGATION_INTERPRETATIONS,
    AGGREGATION_LIST,
    HMR_READS,
    INDEX_DECODES,
    LIVE_STRATEGY_KEY,
    LIVE_STRATEGY_KEYS,
    MODULAR_ACTIVATION_METHODS,
    MODULAR_AGGREGATION_METHODS,
    MODULAR_SCALING_METHODS,
    MODULAR_SPARSITY_METHODS,
    NUM_ACTIVATIONS,
    PALETTE_CONFIGS,
    PALETTE_EVOLUTION_DECODES,
    PALETTE_EVOLUTION_KEYS,
    RANDOM_SCOPES,
    _parse_hmr_reads,
    activation_interpretation_signal,
    aggregation_from_weight_interpretation,
    compute_aggregation_correction_factors,
    continuous_to_index,
    continuous_to_palette_index,
    cppn_output_function_indices,
    dense_substrate_forward,
    grouped_activation_forward,
    grouped_activation_forward_with_palette,
    grouped_aggregation_forward,
    hidden_activation,
    input_drive,
    level_sparsity_quantiles,
    mutate_palette_mask,
    node_sparsity_quantiles,
    palette_snap_table,
    parse_function_palette,
    parse_live_strategy,
    parse_palette_evolution,
    random_function_indices,
    scatter_aggregate_by_target,
    segment_count_2d,
    segment_max_2d,
    segment_min_2d,
    segment_sum_2d,
    sparsify_by_quantile,
    sparsify_winner_take_all,
    stp_depression,
)
from emr_hyperneat.engine.settling import (
    HH_ACTIVATION_MODES,
    _NODE_RULES,
    _SETTLE_STEPS,
    _SETTLE_STEPS_ADDITIVE,
    _SETTLE_STEPS_HMR,
    _STABLE_NODE_FNS,
    _hmr_recurrent_delta,
    _is_stable_node_fn,
    _node_rule,
    _recurrent_input,
    _stable_settle_step,
    _stable_settle_step_additive,
    _stable_settle_step_hmr,
    node_activation,
    node_activation_rule,
    settle_hidden,
    settle_hidden_additive,
    settle_hidden_hmr,
)
from emr_hyperneat.engine.neuromodulation import (
    CUSTOM_LEVEL_TRANSLATIONS,
    HMR_LEVEL_MODES,
    HMR_NAMED_MODE_LEVELS,
    HMR_NEUROMODULATION_MODES,
    HMR_OPTION_B_MODES,
    MODULATION_MODES,
    MODULATORY_BUNDLES,
    MODULATORY_OPTIONS,
    NEUROMODULATION_NODE_RULES,
    NEUROMODULATION_PRESETS,
    NEUROMODULATION_SOURCES,
    NT_PRESETS,
    NT_PRESETS_2,
    NT_PRESETS_3_FIBONACCI,
    NT_PRESETS_4,
    NT_PRESETS_5,
    NT_PRESETS_6,
    NT_TASK_PRESETS,
    NeuromodulationConfig,
    NodeRule,
    _neuromodulated_rule,
    _neuromodulation_block_of,
    _refuse_level_flags_against_named_mode,
    _translate_hmr_neuromodulation,
    apply_gate_scaling,
    check_modulatory_options,
    cppn_modulatory_masks,
    derive_context_from_input,
    derive_receptor_from_weight,
    exact_ceil_counts,
    gated_node_rule,
    get_cppn_num_outputs,
    get_cppn_receptor_base_idx,
    get_neuromodulation_config,
    get_nt_for_task,
    invert_outputs,
    modulatory_cppn_reads,
    modulatory_masks,
    modulatory_neuron_masks,
    modulatory_node_rule,
    neuromodulation_cppn_channels,
    option_b_receptor_outputs,
    resolve_modulatory_options,
    true_neuromodulation_node_fn,
    true_neuromodulation_node_rule,
)
from emr_hyperneat.engine.evaluation import (
    EvaluatedGeneration,
    FeedforwardGeneration,
    _auto_eval_chunk_size,
    _chunked_forward_evaluation,
    _eval_single_fitness,
    _node_list_forward,
    _plain_recurrent_forward,
    _stack_chunks,
    chunked_forward_eval,
    chunked_vmap_eval,
    eval_single_network_hybrid,
    evaluate_population_on_device,
    forward_hybrid_vmapped,
)
from emr_hyperneat.engine.parallel import (
    DeviceSplit,
    ISLAND_KEYS,
    ISLAND_TOPOLOGIES,
    IslandPlan,
    PARALLEL_KEYS,
    SPLIT_STRATEGIES,
    STRATEGY_NAME_MAP,
    evaluate_population_split,
    evaluation_ranges,
    island_seed,
    make_device_split,
    migration_targets,
    parse_islands,
    parse_parallel_grid,
    query_blocks,
    query_population_split,
    remap_hidden_keys,
    split_ranges,
)
from emr_hyperneat.engine.discovery import (
    HHCacheManager,
    SparseCapExceeded,
    SparseCapWarning,
    _KEEP_CEILING,
    _PER_GENOME_BLOCK_PAIRS,
    _PER_GENOME_PAIR_BUDGET,
    _PER_GENOME_ROW_BUDGET,
    _bucket_size_for,
    _constraint_mask_cache,
    _constraint_mask_cache_lock,
    _discover_sparse_hh_multi_hop_single_chunk,
    _discover_sparse_hh_per_genome,
    _discover_sparse_hh_population,
    _extract_sparse_cache,
    _extract_sparse_cache_lock,
    _get_constraint_mask_fn,
    _get_extract_sparse_fn,
    _get_per_genome_discovery_fn,
    _get_sparse_discovery_combine_fn,
    _get_union_multi_hop_fn,
    _last_discovery_stats,
    _per_genome_block_for,
    _per_genome_discovery_cache,
    _per_genome_discovery_cache_lock,
    _per_genome_pair_outputs,
    _per_genome_rows_for,
    _record_discovery_stats,
    _sparse_cap_lock,
    _sparse_cap_run,
    _sparse_cap_run_note,
    _sparse_cap_run_start,
    _sparse_cap_truncation_warned,
    _sparse_discovery_jit_cache,
    _union_multi_hop_cache,
    _union_multi_hop_cache_lock,
    compute_multi_hop_connections_fori_loop,
    apply_sparse_cap_policy,
    compute_multi_hop_connections_matrix_power,
    discover_sparse_hh_vectorized_multi_hop,
    get_connection_constraint_mask_batched,
    grown_sparse_cap,
    hh_pair_weights,
    per_genome_pair_connections,
    sparse_cap_refusal,
    sparse_cap_run_record,
)
from emr_hyperneat.engine.pmap import (
    _PMAP_EVAL_DENSE,
    _PMAP_EVAL_HYBRID,
    _full_pipeline_masks,
    _full_pipeline_masks_jit,
    _full_pipeline_with_hh_core,
    _pmap_data_parallel_full_hh,
    _pmap_eval_batch_dense,
    _pmap_eval_batch_hybrid,
    _pmap_eval_single_dense,
    _pmap_eval_single_hybrid,
)
from emr_hyperneat.engine.hmr_names_mixin import HMRNamesMixin
from emr_hyperneat.engine.palette_mixin import PaletteMixin
from emr_hyperneat.engine.modes_mixin import ModesMixin
from emr_hyperneat.engine.champion_mixin import ChampionMixin
from emr_hyperneat.engine.routes_mixin import RoutesMixin
from emr_hyperneat.engine.multi_device_mixin import MultiDeviceMixin
from emr_hyperneat.engine.multitask_mixin import MultiTaskMixin
from emr_hyperneat.engine.configuration_mixin import ConfigurationMixin
from emr_hyperneat.engine.generation_mixin import GenerationMixin


class EMRHyperNEAT(HMRNamesMixin, PaletteMixin, ModesMixin, ChampionMixin, RoutesMixin,
                   MultiDeviceMixin, MultiTaskMixin, ConfigurationMixin, GenerationMixin,
                   EMRHyperNEATMultiGPU):
    """Unified Extended EMR-HyperNEAT with Dynamic Functions + Neuromodulation.

    This is the full implementation combining:
    - unified_extended.py: All multi-GPU, sparse H->H, multi-hop, caching features
    - dynamic_functions_aggregation.py: Per-node activation/aggregation functions
    - neuromodulation_functions.py: 4 neuromodulation levels + multi-task support

    Features from unified_extended:
    - EMRConfig with all recurrence options + 6 presets
    - SparseHiddenConnections dataclass for type-safe storage
    - h->h discovery per genome or over the population union (hh_discovery_mode)
    - Multi-hop vectorized expansion (JIT-compatible matrix power)
    - Y-coordinate constraint filtering (forward/backward/lateral/self)
    - H->H caching with time-based and change-based refresh
    - Two forward pass modes: dense, and dense W1/W2 with sparse h->h
    - Automatic exhaustive dataset detection for correct multi-GPU routing

    Features from dynamic_functions_aggregation:
    - 18 activation functions (tanh, sigmoid, relu, sin, burst, resonator, etc.)
    - 6 aggregation functions (sum, mean, max, min, product, maxabs)
    - 7 selection modes (disabled, global, cppn_output, weight_interpretation, etc.)
    - Palette system for custom activation function subsets
    - Per-node dynamic function selection

    Features from neuromodulation (4 levels):
    - Level 1: static_gating - one gate in [0, 1] per hidden node from its input weights, or per
      connection from a dedicated CPPN output (gate_source 'cppn')
    - Level 2: context_gating - XdG-style task context modulates gates
    - Level 3: modulatory_neurons - some of a genome's own nodes set a per-case gain for the others
      (activity modulation, not Soltoggio's plasticity gating)
    - Level 4: true_neuromodulation - NT vectors + receptor densities + gain modulation
    - Multi-task support with 6 fitness aggregation methods
    - Neuromodulation applied to both Input->Hidden and H->H iterations

    Dynamic Functions Configuration (a key EMR does not read raises, in these blocks and at the top
    of emr_hyperneat, as does an unknown mode, function name or palette entry):
        dynamic_functions:
          mode: 'disabled'           # disabled, global, cppn_output, weight_interpretation,
                                     # random_fixed, random_generation, modular
          hidden_activation: 'tanh'  # mode 'global' only; defaults to substrate.hidden_activation
          output_activation: 'sigmoid'  # defaults to substrate.output_activation; another function
                                        # raises where the evaluation is plain (no per-node
                                        # function, aggregation or neuromodulation) or multi-task,
                                        # which apply a sigmoid
          num_activations: 4         # without a palette, ACTIVATION_LIST[:num_activations]
          random_scope: 'genome'     # random modes: genome (per slot, per run) or shared (0.1.0's
                                     # draw: one assignment for every genome and run); random_seed
                                     # fixes it
          index_decode: 'tanh'       # tanh (0.1.0's; a palette of 9+ never reaches its end entries)
                                     # or uniform (every entry, equal bins); aggregation has its own
          cppn_read: 'self'          # cppn_output channel at the node's self-connection, or
                                     # input_mean (HMR's)
          weight_source: 'w1'        # interpretations read W1, or raw (HMR's: the unmasked weight
                                     # channel)
          palette: 'sin_only'        # optional: a preset name, or ACTIVATION_LIST indices or names
          interpretation: 'magnitude_bio'  # weight_interpretation: magnitude_bio, variance, sign,
                                           # magnitude, combined, hierarchical_sparsity,
                                           # sparsity_threshold, stp_inspired
          sparsity: {level_0: 0.05, level_1: 0.20, level_2_plus: 0.40}  # fractions kept
          modular:                   # mode 'modular' (HMR's three layers), and aggregation mode
                                     # 'modular'
            activation_method: 'magnitude_bio'  # none (tanh), magnitude_bio, magnitude, sign,
                                                # variance
            sparsity_method: 'none'  # none, hierarchical, threshold, wta (feed-forward
                                     # evaluation only)
            scaling_method: 'none'   # none, stp_inspired
            aggregation_method: 'none'  # none (sum), magnitude_bio, variance, sign_uniformity,
                                        # fan_in
            sparsity: {level_0: 0.05, level_1: 0.20, level_2_plus: 0.40, wta_k_percent: 0.10}
          hh_activation_mode: 'every_iteration'  # every_iteration or tanh (tanh on h->h)

          # CPPN output configuration (for cppn_output mode):
          num_cppn_outputs: 1        # Number of CPPN outputs (1=derive from weights, 2+=dedicated outputs)
          cppn_output_indices:       # Which CPPN output index to use for each purpose
            weight: 0                # Index for weight output (always 0)
            activation: 0            # Index for activation output (0 when num_cppn_outputs=1)
            aggregation: 0           # Index for aggregation output (for future use)

        aggregation:
          mode: 'disabled'           # disabled, global, cppn_output, weight_interpretation,
                                     # random_fixed, random_generation, modular
                                     # (dynamic_functions.modular.aggregation_method)
          global_function: 'sum'     # sum, mean, max, min, product, maxabs
          interpretation: 'magnitude_bio'  # weight_interpretation: magnitude_bio, variance,
                                           # sign_uniformity, fan_in
          num_aggregations: 4
          use_true_aggregation: true
          hh_aggregation_mode: 'sum'  # sum or dynamic

    Neuromodulation Configuration:
        neuromodulation:
          mode: 'disabled'           # disabled, static_gating, context_gating,
                                     # modulatory_neurons, true_neuromodulation

          # Static gating (Level 1)
          gate_threshold: 0.5
          gate_scaling: 'sigmoid'    # sigmoid, binary, soft_threshold; any other name: sigmoid(raw)

          # Context gating - XdG style (Level 2)
          context_dim: 4
          context_influence: 0.5
          context_source: 'task_id'  # the only value: the context is the current task index

          # Modulatory neurons (Level 3; activity modulation, not plasticity)
          mod_neuron_ratio: 0.1      # in (0, 1]
          mod_connection_type: 'multiplicative'  # multiplicative, additive, gated
          mod_decay: 0.9             # scales the mean_decay signal and the variance node weights
          modulatory_source: 'weights'  # the options' bundle: weights or cppn
          # Each option changes one thing; unset, it takes the bundle's value
          modulatory_selection: 'variance'     # variance, cppn (node-type output)
          modulatory_count_rule: 'floor'       # floor, ceil, fixed (with mod_neuron_count)
          modulatory_signal: 'mean_decay'      # mean_decay, sum_factor (mod-factor output)
          modulatory_node_weights: 'variance'  # variance, none
          modulatory_scope: 'all'              # all, standard
          modulatory_cppn_read: 'input_mean'   # input_mean, self
          modulatory_cppn_outputs: 'read'      # read, always
          modulatory_matched_gain: 'none'      # none, mean, shift

          # TRUE neuromodulation (Level 4)
          num_nt_types: 4            # [DA, 5HT, NE, ACh]
          modulation_strength: 2.0
          receptor_from_weight: true # Option A (true) vs B (CPPN outputs)
          receptor_derivation: 'tanh' # tanh, abs, normalized, fourier, softmax,
                                      # orthogonal, phase_shifted
          modulation_mode: 'full'    # full, gating_only, gain_bias_only
          use_output_inversion: true # NT4 controls output inversion

        multitask:
          enabled: false
          num_tasks: 2
          task_names: ['xor', 'and']
          fitness_aggregation: 'mean'  # mean, min, weighted, product, softmin, harmonic
          task_weights: null
          orthogonality_bonus: 0.0
          specialization_bonus: 0.0

    Example with neuromodulation:
        algo = EMRHyperNEAT()
        algo.create_config({
            'algorithm_params': {
                'emrhyperneat': {
                    'emr_hyperneat': {
                        'max_depth': 3,
                        'recurrence': {'preset': 'hidden_only'},
                        'dynamic_functions': {
                            'mode': 'weight_interpretation',
                            'num_activations': 6
                        },
                        'neuromodulation': {
                            'mode': 'true_neuromodulation',
                            'num_nt_types': 4,
                            'receptor_from_weight': True
                        },
                        'multitask': {
                            'enabled': True,
                            'num_tasks': 5,
                            'task_names': ['xor', 'and', 'or', 'nand', 'nor'],
                            'fitness_aggregation': 'min'
                        }
                    }
                }
            }
        })
    """

    #: Accepted values for `dynamic_functions.mode`. `create_config` rejects anything else,
    #: which would otherwise fall through the activation dispatch and silently run the fixed
    #: tanh/sigmoid baseline. 'modular' runs HMR's three layers (the modular block).
    DYNAMIC_FUNCTION_MODES = frozenset({
        'disabled',
        'global',
        'cppn_output',
        'weight_interpretation',
        'random_fixed',
        'random_generation',
        'modular',
    })
    #: Keys of the neuromodulation block EMR reads (_parse_neuromodulation_config). Any other key
    #: raises. `enabled` is read only to refuse a contradiction with the mode.
    NEUROMODULATION_KEYS = frozenset(
        {
            'mode',
            'preset',
            'enabled',
            'gate_threshold',
            'gate_scaling',
            'gate_hardness',
            'context_dim',
            'context_influence',
            'context_source',
            'mod_neuron_ratio',
            'mod_connection_type',
            'mod_decay',
            'num_nt_types',
            'modulation_strength',
            'receptor_from_weight',
            'receptor_derivation',
            'modulation_mode',
            'use_output_inversion',
            'branch_gating',
            'use_self_connection_query',
            'node_rule',
            'gate_source',
            'modulatory_source',
            'gate_from_cppn',
            # The modulatory-neuron options
            *MODULATORY_OPTIONS,
            'mod_neuron_count',
        }
    )
    #: The neuromodulation modes the evaluation dispatches, one level each (context gating carries
    #: its base gates, XdG-style).
    NEUROMODULATION_MODES = ('disabled', 'static_gating', 'context_gating', 'modulatory_neurons',
                             'true_neuromodulation')
    #: Keys of the multitask block (MultiTaskConfig). Any other key raises. Three of these (the two
    #: constant bonuses and confusion_eval_frequency) are accepted only at their default value.
    MULTITASK_KEYS = frozenset({
        'enabled', 'num_tasks', 'task_names', 'nt_per_task', 'fitness_aggregation', 'task_weights',
        'joint_evolution', 'orthogonality_bonus', 'specialization_bonus', 'hidden_activation',
        'per_task_activation', 'fitness_mode', 'modulation_mode', 'modulation_strength_override',
        'two_module_mode', 'generalist_bonus_type', 'generalist_bonus_weight', 'generalist_threshold',
        'softmin_temperature', 'modulation_penalty', 'specialization_bonus_weight',
        'confusion_eval_frequency', 'orthogonality_bonus_weight', 'orthogonality_metric',
        'branch_gating_mode',
    })
    #: Keys of the emr_hyperneat block EMR reads (the base class, this class and
    #: parse_execution_config). Any other key raises.
    EMR_HYPERNEAT_KEYS = frozenset(
        {
            # emrhyperneat_base.py
            'band_threshold',
            'division_threshold',
            'extra_randkey_split',
            'fitness_mode',
            'fuse_w1_computation',
            'initial_depth',
            'iteration_level',
            'max_depth',
            'max_weight',
            'population_chunk_size',
            'skip_unused_masks',
            'sparse_forward_threshold',
            'variance_rule',
            'variance_threshold',
            'verbose',
            # this class
            'aggregation',
            'dynamic_functions',
            'fitness_rescale_variance',
            'geometry_seeding',
            'leo',
            'locality_radius',
            'islands',
            'multi_gpu_strategy',
            'multitask',
            'neuromodulation',
            'parallel',
            'recurrence',
            'streaming_verbose',
            'weight_threshold',
            # execution.parse_execution_config
            'execution',
            'enable_streaming',
        }
    )
    #: Keys of the recurrence block EMR reads (_parse_extended_config). Any other key raises, except
    #: `use_vectorized_discovery`, retired: it was stored and never consulted, so ignoring it
    #: changes no result, and older configurations still carry it (TestRetiredKnobs).
    RECURRENCE_KEYS = frozenset(
        {
            'preset',
            'enabled',
            'allow_hidden_to_hidden',
            'allow_backward',
            'allow_lateral',
            'allow_self_loops',
            'iteration_level',
            'activate_time',
            'max_connections',
            'max_sparse_conns',
            'multi_hop_algorithm',
            'hop_decay_factor',
            'multi_hop_second_factor',
            'hh_cache_enabled',
            'hh_refresh_interval',
            'hh_mask_change_threshold',
            'use_dense_discovery',
            'hh_discovery_mode',
            'eval_chunk_size',
            'use_vectorized_discovery',
            'settling_rule',
            'hh_cache_trigger',
            'network_model',
            'sparse_cap_policy',
            'max_sparse_conns_ceiling',
        }
    )
    #: Keys of the recurrence block the grid model alone reads: under network_model 'node_list'
    #: each raises when present. hh_cache_enabled and settling_rule raise only at a value the node
    #: list does not take (true, 'hmr'). max_connections is HMR's node-list cap, which nothing read
    #: either (HMR padded to 500 whatever it said).
    NODE_LIST_REFUSED_RECURRENCE_KEYS = (
        'max_connections', 'max_sparse_conns', 'multi_hop_algorithm', 'hop_decay_factor',
        'multi_hop_second_factor', 'hh_refresh_interval', 'hh_mask_change_threshold',
        'hh_cache_trigger', 'use_dense_discovery', 'hh_discovery_mode',
        'sparse_cap_policy', 'max_sparse_conns_ceiling',
    )
    #: Keys of the leo and geometry_seeding blocks EMR reads; any other key raises (a misspelled key
    #: would leave the run on the default). The execution block's keys are EXECUTION_KEYS.
    LEO_KEYS = frozenset({'enabled', 'threshold', 'mode', 'output_index'})
    #: Keys of the dynamic_functions block EMR reads (_parse_dynamic_functions_config). Any other key
    #: raises rather than leaving a run without what its configuration asks for (HMR's aggregation
    #: runners nest `aggregation` here; others set `activation_palette`). `palette_evolution` is
    #: the built-in palette evolution (:func:`parse_palette_evolution`), `live_strategy` HMR's live
    #: palette strategy (:func:`parse_live_strategy`).
    DYNAMIC_FUNCTIONS_KEYS = frozenset(
        {
            'mode',
            'random_seed',
            'hidden_activation',
            'output_activation',
            'num_activations',
            'hh_activation_mode',
            'palette',
            'interpretation',
            'sparsity',
            'critical_periods',
            'modular',
            'cppn_output_indices',
            'num_cppn_outputs',
            'palette_evolution',
            'random_scope',
            'index_decode',
            'cppn_read',
            'weight_source',
            'live_strategy',
        }
    )
    #: Keys of the blocks nested in dynamic_functions.
    CPPN_OUTPUT_INDICES_KEYS = frozenset({'weight', 'activation', 'aggregation'})
    SPARSITY_KEYS = frozenset({'level_0', 'level_1', 'level_2_plus', 'wta_k_percent'})
    CRITICAL_PERIODS_KEYS = frozenset({'enabled', 'phase1_end', 'phase2_end', 'min_plasticity'})
    #: Keys of dynamic_functions.modular and of the two sparsity blocks (dynamic_functions.sparsity
    #: and dynamic_functions.modular.sparsity).
    MODULAR_KEYS = frozenset({'activation_method', 'sparsity_method', 'scaling_method',
                              'aggregation_method', 'sparsity'})
    #: Keys of the aggregation block EMR reads; any other key raises.
    AGGREGATION_KEYS = frozenset(
        {
            'mode',
            'random_seed',
            'global_function',
            'num_aggregations',
            'interpretation',
            'use_true_aggregation',
            'hh_aggregation_mode',
            'random_scope',
            'palette',
            'index_decode',
            'cppn_read',
            'weight_source',
        }
    )
    #: Accepted values for `aggregation.mode`; anything else raises rather than running sum.
    AGGREGATION_MODES = frozenset({
        'disabled', 'global', 'cppn_output', 'weight_interpretation', 'random_fixed', 'random_generation',
        'modular',
    })
    #: Keys of the geometry_seeding block EMR reads; any other key raises.
    GEOMETRY_SEEDING_KEYS = frozenset({'enabled', 'seed_weight'})
    #: Keys of the levels outside emr_hyperneat, as the base class's create_config reads them. Any
    #: other key raises (_refuse_unknown_outer_keys).
    #: The top level of a nested configuration:
    TOP_LEVEL_KEYS = frozenset({'algorithm_params', 'config_file', 'preset', 'overrides'})
    ALGORITHM_PARAMS_KEYS = frozenset({'emrhyperneat'})
    #: algorithm_params.emrhyperneat, or the top level of a flat configuration.
    ALGORITHM_PARAMS_EMRHYPERNEAT_KEYS = frozenset({
        'emr_hyperneat', 'substrate', 'swarm', 'neat', 'population_size', 'cppn', 'neat_mutation',
        'neat_species', 'neat_selection',
    })
    SUBSTRATE_KEYS = frozenset({'input_coords', 'output_coords', 'activate_time',
                                'hidden_activation', 'output_activation'})
    SWARM_KEYS = frozenset({'enabled', 'num_agents', 'team_layout'})
    #: `species_size` is read by nothing and accepted without a warning: the per-node paper's
    #: runners carry it (docs/configuration.md, "Keys that parse but do not act").
    NEAT_KEYS = frozenset({'pop_size', 'population_size', 'species_size'})
    CPPN_KEYS = frozenset({'genome', 'activation'})
    CPPN_GENOME_KEYS = frozenset({'num_inputs', 'num_outputs'})
    CPPN_ACTIVATION_KEYS = frozenset({'options', 'default', 'mutate_rate'})

    def __init__(
        self,
        name: str = 'emrhyperneat-unified-extended-dynamic-functions',
        implementation: str = 'tensorneat-emrhyperneat-unified-extended-dynamic-functions',
        strategy: MultiGPUStrategy = MultiGPUStrategy.SINGLE_GPU,
        position_config: Optional[Any] = None,
        island_config: Optional[Any] = None,
        hybrid_config: Optional[Any] = None,
        pmap_config: Optional[Any] = None,
    ):
        """Initialize unified extended EMR-HyperNEAT.

        Args:
            name: Algorithm name
            implementation: Implementation identifier
            strategy: GPU execution strategy, a name of STRATEGY_NAME_MAP or SINGLE_GPU_PYTHON_LOOP
            position_config: Refused when given: it configured the base class's own
                implementation of a strategy, which this package no longer has (the split routes
                read emr_hyperneat.parallel, the island model emr_hyperneat.islands).
            island_config: Refused when given, as position_config.
            hybrid_config: Refused when given, as position_config.
            pmap_config: Refused when given, as position_config.

        Raises:
            ValueError: A strategy name without a route, or any of the four sharding configurations.
        """
        # The configuration key's meaning (STRATEGY_NAME_MAP); the Python-loop name keeps the
        # one-device loop. The base class is set up for one device, as when the configuration key
        # chooses the strategy.
        if strategy == MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP:
            routed = strategy
        elif strategy.name in STRATEGY_NAME_MAP:
            routed = STRATEGY_NAME_MAP[strategy.name]
        else:
            raise ValueError(
                f"strategy {strategy.name} has no route in EMRHyperNEAT; the names are "
                f"{sorted(STRATEGY_NAME_MAP)} (and SINGLE_GPU_PYTHON_LOOP)")
        # The four sharding configurations configured the base class's own implementations of the
        # HMR engine's strategies, which this package no longer has, so nothing would read them:
        # they raise.
        given = [key for key, value in (('position_config', position_config),
                                        ('island_config', island_config),
                                        ('hybrid_config', hybrid_config),
                                        ('pmap_config', pmap_config))
                 if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} configure base-class strategies, the base "
                             f"class's own implementations, which EMRHyperNEAT does not run; "
                             f"{routed.name} reads none of them (the split routes read "
                             f"emr_hyperneat.parallel, the island model emr_hyperneat.islands)")
        super().__init__(
            name=name,
            implementation=implementation,
            strategy=MultiGPUStrategy.SINGLE_GPU,
        )
        self.strategy = routed

        # Extended configuration
        self.extended_config: Optional[EMRConfig] = None
        # An explicit substrate.activate_time from the last create_config (None: the key was absent), which the
        # settling steps honor there and in a later set_recurrence_config (resolve_recurrence)
        self._substrate_activate_time: Any = None

        # Cache manager
        self._hh_cache: Optional[HHCacheManager] = None

        # The devices a split route spreads the generation's query and evaluation over; None runs
        # both on one device, as every route but the split ones does
        self._device_split: Optional[DeviceSplit] = None
        self._parallel_grid: Optional[Tuple[Optional[int], Optional[int]]] = None
        # The island model: its islands (parse_islands at create_config), the
        # configuration each island's engine is built from, and the run's seed (initialize's), from
        # which every island's derives
        self._island_plan: Optional[IslandPlan] = None
        self._island_source: Optional[Dict[str, Any]] = None
        self._run_seed: Optional[int] = None
        # What a generation's tell receives instead of its own state and fitness (_before_tell): the
        # island model's migration. None: the generation's own
        self._pre_tell_hook: Optional[Callable[[Any, jnp.ndarray], Tuple[Any, jnp.ndarray]]] = None

        # Metrics tracking
        self._extended_metrics: Optional[EMRRecurrenceMetrics] = None
        # execution.generation_loop 'while_loop': the compiled loop per history capacity
        # (max_generations), cleared where the configuration it read can change (create_config,
        # initialize, _recompile_tell)
        self._generation_loop_cache: Dict[int, Callable] = {}

        # Forward pass mode
        self._forward_mode: ForwardPassMode = ForwardPassMode.DENSE_ONLY

        # Current generation (for caching)
        self._current_generation: int = 0

        # Fitness rescaling: when set, converts fitness from 1-MSE to R²
        # R² = 1 - MSE/target_variance, spreading the useful range
        self._fitness_rescale_variance: Optional[float] = None

        # The input-to-hidden and hidden-to-output pruning threshold
        # (emr_hyperneat.weight_threshold)
        self.weight_threshold: float = WEIGHT_THRESHOLD

        # HMR's live palette strategy: the checked block, the runner's strategy object
        # (set_palette_strategy), and the generations and best fitnesses the object has been told
        self._live_strategy: Optional[Dict[str, Any]] = None
        self._palette_strategy: Any = None
        self._live_generation: int = 0
        self._live_fitness_history: List[float] = []

        # Dynamic Functions Configuration (from dynamic_functions_aggregation.py)

        # Activation function settings
        self.dynamic_functions_mode: str = 'disabled'  # disabled, global, cppn_output, weight_interpretation, random_fixed, random_generation, modular
        self.df_hidden_activation: str = 'tanh'
        self.df_output_activation: str = 'sigmoid'
        self.df_num_activations: int = 4
        self.df_palette: Optional[jnp.ndarray] = None  # Custom palette indices
        self.df_palette_name: Optional[str] = None  # Named palette
        self.df_interpretation: str = 'magnitude_bio'  # For weight_interpretation mode
        self.hh_activation_mode: str = 'initial_only'  # initial_only or every_iteration
        self._settling_rule: str = 'emr'  # recurrence.settling_rule (SETTLING_RULES)
        # recurrence.network_model (NETWORK_MODELS), the node list's parsed settings, and its last
        # networks
        self.network_model: str = 'grid'
        self._node_list_settings: Optional[node_list.NodeListSettings] = None
        self._cached_node_list: Optional[node_list.NodeListNetworks] = None

        # Aggregation function settings
        self.agg_mode: str = 'disabled'  # disabled, global, cppn_output, weight_interpretation
        self.agg_global_function: str = 'sum'
        self.agg_num_aggregations: int = 4
        self.agg_use_true_aggregation: bool = True  # True = accurate, False = approximation
        self.agg_interpretation: str = 'magnitude_bio'
        # AGGREGATION_LIST indices the per-node modes choose among (aggregation.palette; a runner
        # may also set it between generations, as for df_palette). None:
        # AGGREGATION_LIST[:agg_num_aggregations].
        self.agg_palette: Optional[jnp.ndarray] = None
        # How channels and weight-interpretation signals become indices (INDEX_DECODES)
        self.df_index_decode: str = 'tanh'
        self.agg_index_decode: str = 'tanh'
        # Where cppn_output reads its channel and what the weight interpretations read (HMR_READS)
        self.df_cppn_read: str = 'self'
        self.df_weight_source: str = 'w1'
        self.agg_cppn_read: str = 'self'
        self.agg_weight_source: str = 'w1'
        # hh_aggregation_mode controls H->H aggregation behavior:
        # - 'sum': Fast scatter_add (default, best performance)
        # - 'dynamic': Per-node aggregation using segment operations
        self.hh_aggregation_mode: str = 'sum'  # sum or dynamic

        # Sparsity settings (from dynamic_functions_aggregation.py)
        self.df_sparsity_config: Dict[str, float] = {
            'level_0': 0.05,  # DG-like, 5% active
            'level_1': 0.20,  # CA3-like, 20% active
            'level_2_plus': 0.40,  # CA1-like, 40% active
        }

        # Cached activation/aggregation functions
        self._hidden_act_fn: Optional[Callable] = None
        self._output_act_fn: Optional[Callable] = None

        # CPPN output configuration for cppn_output mode
        # num_cppn_outputs: How many outputs the CPPN has (default 1)
        # When num_cppn_outputs=1, we use the same weight output for activation derivation
        # When num_cppn_outputs>1, each output can be dedicated (weight, activation, aggregation)
        self.num_cppn_outputs: int = 1
        self._cppn_output_indices: Dict[str, int] = {
            'weight': 0,
            'activation': 0,  # Default to same as weight when num_cppn_outputs=1
            'aggregation': 0,
        }
        # The last generation's champion as evaluated, for evaluate_on_data.
        self._last_eval: Optional[Dict[str, Any]] = None

        # Random mode state
        self._random_mode_seed: Optional[int] = None
        self._random_fixed_indices: Optional[jnp.ndarray] = None
        self._random_agg_seed: Optional[int] = None
        self._random_fixed_agg_indices: Optional[jnp.ndarray] = None

        # Modular mode configuration
        self.df_modular_config: Dict[str, str] = {
            'activation_method': 'magnitude_bio',
            'sparsity_method': 'none',
            'scaling_method': 'none',
            'aggregation_method': 'none',
        }
        self.df_modular_sparsity: Dict[str, float] = {
            'level_0': 0.05,
            'level_1': 0.20,
            'level_2_plus': 0.40,
            'wta_k_percent': 0.10,
        }

        # Critical periods configuration
        self.df_critical_periods_enabled: bool = False
        self.df_critical_periods_config: Dict[str, float] = {
            'phase1_end': 0.2,
            'phase2_end': 0.5,
            'min_plasticity': 0.3,
        }

        # Neuromodulation Configuration (4 levels supported)
        # Level 1: static_gating - one gate in [0, 1] per hidden node from its input weights, or
        # per connection from a dedicated CPPN output (gate_source 'cppn')
        # Level 2: context_gating - XdG-style context modulates gates
        # Level 3: modulatory_neurons - some of a genome's own nodes set a per-case gain for the
        # others
        # Level 4: true_neuromodulation - NT vectors + receptor densities + gain modulation

        # Neuromodulation config (parsed from user config in create_config)
        self.neuromod_config: Optional[NeuromodulationConfig] = None

        # TRUE neuromodulation storage (populated during _process_cppn_outputs)
        # - receptor_densities: (pop_size, total_positions, num_nt_types)
        # - base_gains: (pop_size, total_positions)
        # - modulation_strength: float
        # - option: 'A' (from weight) or 'B' (dedicated CPPN outputs)
        self._neuromod_true: Dict[str, Any] = {
            'receptor_densities': None,
            'base_gains': None,
            'num_nt_types': 4,
            'modulation_strength': 2.0,
            'option': 'A',  # Default: derive from weights
        }

        # Modulatory neurons storage (activity modulation, not plasticity)
        self._mod_neuron_mask: Optional[jnp.ndarray] = None  # Boolean mask of modulatory neurons
        self._mod_connection_weights: Optional[jnp.ndarray] = None  # Modulatory connections

        # Multi-Task Configuration

        # Multi-task config (parsed from user config in create_config)
        self.multitask_config: Optional[MultiTaskConfig] = None

        # Task-specific neurotransmitter vectors
        # - nt_vectors: (num_tasks, num_nt_types)
        # - task_fitnesses: (pop_size, num_tasks)
        self._multitask_state: Dict[str, Any] = {
            'nt_vectors': None,
            'task_fitnesses': None,
            'task_names': [],
            'current_task_idx': 0,
        }

        # Level-by-Level Streaming Configuration
        # When enabled, CPPN queries are done one hierarchical level at a time
        # instead of all positions at once: one level's queries, and the next level's input
        # queries for its variance, are resident at once; W1 and W2 are still assembled at
        # full size.

        self.enable_streaming: bool = False  # Default: standard
        self._streaming_verbose: bool = False  # Per-level timing if True

        # Execution mode (memory residency)
        # Where a generation's intermediate arrays live: all on device
        # (standard), one level at a time (streaming), or assembled in host RAM
        # and moved to the device chunk by chunk (depth_offload). Defaults to
        # standard, so behavior is unchanged unless a config asks otherwise.
        #
        # Orthogonal to MultiGPUStrategy, which is a device-parallelism axis.
        self.execution_config: ExecutionConfig = ExecutionConfig()
        self._apply_execution_config()

        # Depth-offload runtime counters, read back through get_memory_stats().
        self._cpu_memory_peak_gb: float = 0.0
        self._gpu_transfers_count: int = 0

        # LEO Configuration (Link Expression Output)
        # LEO adds a second CPPN output that gates connection existence
        # independently from weight magnitude. This enables better modular
        # network formation by separating "should this connection exist?"
        # from "how strong should it be?"
        #
        # Connection exists only if BOTH conditions are met:
        #   1. |weight| > weight_threshold
        #   2. leo_output > leo_threshold
        #
        # Expected improvement: +27 percentage points on modular problems
        # (30% -> 57% solve rate on retina, Verbancsics & Stanley 2011)
        self.use_leo: bool = False  # Disabled by default (opt-in feature)
        self.leo_threshold: float = 0.2  # LEO output must exceed this to create connection
        self.leo_mode: str = 'compound'  # see LEO_MODES

    def _refuse_unknown_outer_keys(self, params: Dict[str, Any],
                                   algo_params: Dict[str, Any]) -> None:
        """Raise on a key outside emr_hyperneat that no read site reads.

        The base class's create_config reads the top level, algorithm_params,
        algorithm_params.emrhyperneat (the top level of a flat configuration), substrate, swarm,
        neat and cppn with its genome and activation blocks, without a check, so a misspelled or
        misplaced key there would be ignored without a word. Such a key raises, naming the right
        key when it is known, as keys inside emr_hyperneat do (`_refuse_unknown_keys`) and those of
        the neat_* blocks (`_neat_operator_block`). 0.1.0 ignored it without a word.

        Nothing is checked without an emr_hyperneat block: the base class then warns that every
        setting takes its default, or refuses an HMR-shaped configuration. A ``preset``,
        ``config_file`` or ``overrides`` with a value never gets here (`refuse_file_configuration`).

        Args:
            params: The configuration passed to `create_config`.
            algo_params: Its algorithm_params.emrhyperneat, or ``params`` itself in the flat form.

        Raises:
            ValueError: A key outside emr_hyperneat that EMR does not read.
        """
        emr_block = algo_params.get('emr_hyperneat')
        if not isinstance(emr_block, dict) or not emr_block:
            return
        if algo_params is params:
            flat_keys = self.ALGORITHM_PARAMS_EMRHYPERNEAT_KEYS | self.TOP_LEVEL_KEYS
            _refuse_unknown_keys('', params, flat_keys)
        else:
            _refuse_unknown_keys('', params, self.TOP_LEVEL_KEYS)
            _refuse_unknown_keys('algorithm_params.emrhyperneat', algo_params,
                                 self.ALGORITHM_PARAMS_EMRHYPERNEAT_KEYS)
        _refuse_unknown_keys('algorithm_params', params.get('algorithm_params'),
                             self.ALGORITHM_PARAMS_KEYS)
        _refuse_unknown_keys('substrate', algo_params.get('substrate'), self.SUBSTRATE_KEYS)
        _refuse_unknown_keys('swarm', algo_params.get('swarm'), self.SWARM_KEYS)
        _refuse_unknown_keys('neat', algo_params.get('neat'), self.NEAT_KEYS)
        cppn = algo_params.get('cppn')
        _refuse_unknown_keys('cppn', cppn, self.CPPN_KEYS)
        if isinstance(cppn, dict):
            _refuse_unknown_keys('cppn.genome', cppn.get('genome'), self.CPPN_GENOME_KEYS)
            _refuse_unknown_keys('cppn.activation', cppn.get('activation'),
                                 self.CPPN_ACTIVATION_KEYS)

    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create configuration with extended recurrence support.

        Args:
            params: Configuration parameters

        Returns:
            Configuration object
        """
        # A configuration to load from a file never loaded: no such file ships
        refuse_file_configuration(params)
        # The configuration as given, the island model's source (create_config injects CPPN settings
        # into a copy)
        given_params = params
        # The NEAT algorithm is rebuilt with the configured mutation rates: a topology freeze ends
        # here
        self._frozen_structural_rates = None
        # A compiled generation loop read the previous configuration
        self._generation_loop_cache = {}
        # Parse extended configuration BEFORE calling parent
        algo_params = params.get('algorithm_params', {}).get('emrhyperneat', params)
        hmr_config = algo_params.get('emr_hyperneat', {})
        # A missing block is warned about once, in the base create_config that
        # `super()` reaches below; warning here too would double it.
        _refuse_unknown_keys('emr_hyperneat', hmr_config, self.EMR_HYPERNEAT_KEYS)
        self._refuse_unknown_outer_keys(params, algo_params)
        if hmr_config.get('iteration_level', 1) != 1:
            raise ValueError(
                "emr_hyperneat.iteration_level is not read (the multi-hop level is "
                "recurrence.iteration_level); 0.1.0 ignored a value other than 1, with a "
                "deprecation warning")

        # Opt-in reproduction flag (default False -> EMR's existing behavior is unchanged).
        # When True, EMR performs HMR's extra per-generation pre-ask jax.random.split.
        # HMR's default (extra_randkey_split=True) produced the published results, so this
        # lets reproduction benchmarks recover HMR's exact per-seed trajectories in EMR.
        # Stored under a distinct name so the base class's own `extra_randkey_split`
        # attribute (which defaults True) cannot leak in; every route of run_until_threshold
        # reads this one (0.1.0's FULL_PIPELINE_PARALLEL read the base attribute).
        self._repro_extra_randkey_split = hmr_config.get('extra_randkey_split', False)

        # Check if geometry seeding is enabled - need to set CPPN num_inputs to 7
        geo_config = hmr_config.get('geometry_seeding', {})
        _refuse_unknown_keys('geometry_seeding', geo_config, self.GEOMETRY_SEEDING_KEYS)
        _refuse_unknown_keys('leo', hmr_config.get('leo', {}), self.LEO_KEYS)
        _refuse_unknown_keys('execution', hmr_config.get('execution') or {}, EXECUTION_KEYS)
        geo_seeding_enabled = geo_config.get('enabled', False)

        # Auto-compute required CPPN outputs from dynamic_functions + aggregation config
        dynamic_funcs = hmr_config.get('dynamic_functions', {})
        agg_config_section = hmr_config.get('aggregation', {})
        required_outputs = 1  # weight always needs output 0
        if hmr_config.get('leo', {}).get('enabled', False):
            # LEO's expression output, channel LEO_CHANNEL, counted: otherwise a LEO-only
            # configuration would build a one-output CPPN, and reading channel 1 of it would
            # return the weight (JAX clamps out-of-range indices).
            required_outputs += 1
        if dynamic_funcs.get('mode') == 'cppn_output':
            required_outputs += 1  # activation needs its own output
        if agg_config_section.get('mode') == 'cppn_output':
            required_outputs += 1  # aggregation needs its own output
        # Neuromodulation's dedicated outputs (Option B's receptors, a CPPN gate, the modulatory
        # factor and node type), after every other channel
        required_outputs += len(
            neuromodulation_cppn_channels(hmr_config.get('neuromodulation', {}))
        )

        # Allow explicit override (for ablation configs like 2-output variants)
        explicit_num = dynamic_funcs.get('num_cppn_outputs', None)
        if explicit_num is not None:
            required_outputs = explicit_num

        # Determine if we need to deepcopy and inject CPPN genome params
        needs_deepcopy = geo_seeding_enabled or required_outputs > 1

        if needs_deepcopy:
            params = copy.deepcopy(params)

            # Create the missing levels of the cppn.genome path
            if 'algorithm_params' not in params:
                params['algorithm_params'] = {}
            if 'emrhyperneat' not in params['algorithm_params']:
                params['algorithm_params']['emrhyperneat'] = {}
            if 'cppn' not in params['algorithm_params']['emrhyperneat']:
                params['algorithm_params']['emrhyperneat']['cppn'] = {}
            if 'genome' not in params['algorithm_params']['emrhyperneat']['cppn']:
                params['algorithm_params']['emrhyperneat']['cppn']['genome'] = {}

            cppn_genome = params['algorithm_params']['emrhyperneat']['cppn']['genome']

            if geo_seeding_enabled:
                # Geometry seeding requires 7 CPPN inputs: [x1, y1, x2, y2, delta_x, delta_y, bias]
                cppn_genome['num_inputs'] = 7
                if self.verbose:
                    print("[EMR-HyperNEAT] Geometry seeding: CPPN num_inputs set to 7 [x1, y1, x2, y2, delta_x, delta_y, bias]")

            if required_outputs > 1:
                # Inject num_outputs so NEAT genome is built with correct output count
                cppn_genome['num_outputs'] = required_outputs
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Auto-computed CPPN num_outputs={required_outputs} "
                          f"(act_mode={dynamic_funcs.get('mode', 'disabled')}, "
                          f"agg_mode={agg_config_section.get('mode', 'disabled')})")

        # Store for later use by _parse_dynamic_functions_config
        self._auto_num_cppn_outputs = required_outputs

        # Parse fitness rescaling (converts 1-MSE fitness to R² for compressed landscapes)
        rescale_var = hmr_config.get('fitness_rescale_variance', None)
        if rescale_var is not None and not rescale_var > 0:
            # A variance that is not positive raises rather than leaving the run on 1-MSE
            raise ValueError(f"emr_hyperneat.fitness_rescale_variance must be a positive variance, got "
                             f"{rescale_var!r}")
        if rescale_var is not None and rescale_var > 0:
            self._fitness_rescale_variance = float(rescale_var)
            if self.verbose:
                print(f"[EMR-HyperNEAT] Fitness rescaling enabled: "
                      f"target_variance={rescale_var:.6f}, "
                      f"fitness = R² = 1 - MSE/var")

        # Call parent to set up base configuration
        config = super().create_config(params)
        if self.fitness_mode not in FITNESS_MODES:
            raise ValueError(f"emr_hyperneat.fitness_mode must be one of {FITNESS_MODES}, got "
                             f"{self.fitness_mode!r}; 0.1.0 scored an unknown name as MSE")
        # The input-to-hidden and hidden-to-output pruning threshold: WEIGHT_THRESHOLD unless set.
        # HMR's unified model pruned at band_threshold in its hybrid mode, which its recipe sets
        # here.
        weight_threshold = hmr_config.get('weight_threshold', WEIGHT_THRESHOLD)
        if isinstance(weight_threshold, bool) or not isinstance(weight_threshold, numbers.Real) \
                or not 0.0 <= weight_threshold < self.max_weight:
            raise ValueError(
                "emr_hyperneat.weight_threshold must be a number in [0, "
                f"max_weight={self.max_weight}), "
                f"got {weight_threshold!r}"
            )
        self.weight_threshold = float(weight_threshold)
        # HMR's base-engine memory and speed settings: checked, and inert here. The base class
        # parses them and only its own generation reads them; this class computes the
        # reached-position mask alone (what skip_unused_masks selected) and narrows the evaluation
        # by compaction, so no value changes a result (tests/test_hmr_base_settings.py).
        for key in ('fuse_w1_computation', 'skip_unused_masks'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"emr_hyperneat.{key} must be a bool, got {getattr(self, key)!r}")
        threshold = self.sparse_forward_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, numbers.Integral)
            or threshold < -1
        ):
            raise ValueError(
                "emr_hyperneat.sparse_forward_threshold must be an integer >= -1 (-1 dense, 0 "
                f"sparse, above 0 sparse above that many positions), got {threshold!r}"
            )
        recurrence_section = hmr_config.get('recurrence', {})

        # Parse multi_gpu_strategy from config (string -> enum)
        strategy_str = hmr_config.get('multi_gpu_strategy', None)
        if strategy_str is not None:
            strategy_str_upper = strategy_str.upper()
            # 5 available GPU strategies
            # Decision tree:
            #   Dataset fits on single GPU?
            #   |- yes -> SINGLE_GPU
            #   `- no  -> Multiple GPUs available?
            #            |- no  -> none (STREAMING splits the CPPN queries by level, not the data)
            #            `- yes -> Recurrent/h->h mode?
            #                     |- yes -> EVAL_ONLY_PARALLEL (any iteration level)
            #                     `- no  -> FULL_PIPELINE_PARALLEL (data parallel)
            #
            # POPULATION_PARALLEL_PROCESS: not recommended (see MultiGPUStrategy)
            strategy_map = STRATEGY_NAME_MAP
            if strategy_str_upper in strategy_map:
                self.strategy = strategy_map[strategy_str_upper]
                if self.verbose:
                    print(f"[EMR-HyperNEAT] Strategy from config: {strategy_str} -> "
                          f"{self.strategy}")
            else:
                # An unknown name raises rather than running the default strategy.
                raise ValueError(
                    f"Unknown multi_gpu_strategy {strategy_str!r}; the names are "
                    f"{sorted(strategy_map)}")
        # The split routes' device grid, checked against the strategy the configuration ends with
        self._parallel_grid = parse_parallel_grid(hmr_config.get('parallel'), self.strategy)
        # The island model's islands, checked against the population and the elitism the
        # base class built; each island's engine is built from this configuration, its population
        # and strategy replaced
        self._island_plan = parse_islands(
            hmr_config.get('islands'),
            self.strategy,
            int(self.population_size),
            int(self.neat_algo.species_controller.genome_elitism),
            len(jax.devices()),
        )
        self._island_source = copy.deepcopy(given_params) if self._island_plan is not None else None

        # The base class read substrate.activate_time into activate_time; whether it was set decides the settling
        # steps of a recurrent configuration, here and in a later set_recurrence_config
        self._substrate_activate_time = algo_params.get('substrate', {}).get('activate_time')
        self._parse_extended_config(recurrence_section)

        # LEO (Link Expression Output): a CPPN output that gates whether a connection is
        # expressed, read from channel LEO_CHANNEL. Parsed before the dynamic functions,
        # whose per-node outputs take the channels after it.
        leo_config = hmr_config.get('leo', {})
        self.use_leo = bool(leo_config.get('enabled', False))
        self.leo_threshold = float(leo_config.get('threshold', 0.2))
        self.leo_mode = leo_config.get('mode', 'compound')
        if self.leo_mode not in LEO_MODES:
            raise ValueError(
                f"leo.mode must be one of {LEO_MODES}, got {self.leo_mode!r}"
            )
        if leo_config.get('output_index', LEO_CHANNEL) != LEO_CHANNEL:
            raise ValueError(
                f"leo.output_index must be {LEO_CHANNEL}: LEO is read from that CPPN output "
                "on every path"
            )
        if (
            self.use_leo
            and self.leo_mode == 'hmr_compound'
            and hmr_config.get('locality_radius') is not None
        ):
            # HMR applied the locality penalty to the scaled weight, EMR's LEO paths to the
            # tanh-normalized one; no runner combined them
            raise ValueError(
                "leo.mode 'hmr_compound' with locality_radius is refused: HMR applied the penalty "
                "after scaling the weight, EMR's LEO paths before"
            )
        if self.use_leo and self.verbose:
            print(f"[EMR-HyperNEAT] LEO ENABLED (threshold={self.leo_threshold}, "
                  f"mode={self.leo_mode}, channel {LEO_CHANNEL})")

        # Parse dynamic functions configuration
        self._parse_dynamic_functions_config(hmr_config)

        # Parse neuromodulation configuration (4 levels + multi-task)
        self._parse_neuromodulation_config(hmr_config)

        # Parse streaming configuration
        # Parse execution mode (memory residency). `enable_streaming: true` is
        # still accepted and means `mode: streaming`; an explicit `execution.mode`
        # wins over it.
        # Updates the existing object rather than replacing it, so options set
        # imperatively before create_config survive the parse.
        self.execution_config = parse_execution_config(
            hmr_config, base=self.execution_config
        )
        self._apply_execution_config()
        # The generation loop selects its build step from this flag, so resolve
        # AUTO here where max_depth is known. `run_until_threshold` also flips it
        # for the duration of a MultiGPUStrategy.STREAMING run and restores it in a
        # finally, which keeps working. (Not an OOM fallback, as this comment once
        # claimed: there is no OOM handler on this path, and nothing else in the
        # package writes the flag.)
        self.enable_streaming = (
            self.execution_config.resolve(self.max_depth) is ExecutionMode.STREAMING
        )
        self._streaming_verbose = hmr_config.get('streaming_verbose', False)
        # variance_rule applies on every execution mode, including one chosen at run time: streaming
        # and depth_offload take the legacy level-0 rule through legacy_level0_variance.

        # Parse locality radius for geometry seeding (distance-based weight penalty)
        # None = no penalty (default), positive float = penalty for distances > radius
        self.locality_radius = hmr_config.get('locality_radius', None)
        if self.locality_radius is not None and self.verbose:
            print(f"[EMR-HyperNEAT] Locality radius enabled: {self.locality_radius}")

        # Parse geometry seeding configuration (Risi & Stanley 2012 paper approach)
        # This adds distance as an explicit CPPN input and seeds initial topology
        geo_config = hmr_config.get('geometry_seeding', {})
        self.geometry_seeding_enabled = geo_config.get('enabled', False)
        self.geometry_seeding_weight = geo_config.get('seed_weight', -1.0)
        if self.geometry_seeding_enabled and self.verbose:
            print(f"[EMR-HyperNEAT] Geometry seeding enabled (seed_weight={self.geometry_seeding_weight})")
            print("[EMR-HyperNEAT] CPPN inputs: [x1, y1, x2, y2, delta_x, delta_y, bias] (7 dimensions)")

        # Log swarm mode status
        if self.swarm_mode and self.verbose:
            cppn_dims = 7 + (2 if self.geometry_seeding_enabled else 0)
            print(f"[EMR-HyperNEAT] Swarm mode enabled: {self.swarm_num_agents} agents, "
                  f"layout={self.swarm_team_layout}")
            print(f"[EMR-HyperNEAT] CPPN inputs: [agent_x, agent_y, x1, y1, x2, y2, bias] "
                  f"({cppn_dims} dimensions)")

        # Initialize cache manager
        if self.extended_config is not None:
            self._hh_cache = HHCacheManager(self.extended_config)

        # The node-list model, checked once everything above is parsed
        self._node_list_settings = None
        self._cached_node_list = None
        if self.network_model == 'node_list':
            self._configure_node_list(hmr_config, algo_params, recurrence_section)

        # The island model's migration, checked once everything above is parsed
        if self._island_plan is not None and self._island_plan.migration_interval > 0:
            self._refuse_migration_gaps()

        self._check_feature_combinations()
        return config

    def _configure_node_list(self, hmr_config: Dict[str, Any], algo_params: Dict[str, Any],
                             recurrence_section: Dict[str, Any]) -> None:
        """Check a node-list configuration and keep its settings.

        The node list (node_list.py) reads the depth, the band threshold, max_weight, the
        connection classes, iteration_level as its phase-2 rounds, the settling steps,
        population_chunk_size and the evaluation's own settings. A setting it cannot apply raises,
        every one in one message: the grid's variance and pruning, the grid-only recurrence keys
        (NODE_LIST_REFUSED_RECURRENCE_KEYS), per-node functions, aggregation, neuromodulation,
        multi-task evaluation, LEO, geometry seeding, locality, swarm mode, a CPPN of another
        shape, streaming, depth offload, the multi-device strategies and a subclass that replaces
        the generation.

        Args:
            hmr_config: The emr_hyperneat block.
            algo_params: algorithm_params.emrhyperneat.
            recurrence_section: The recurrence block.

        Raises:
            ValueError: A setting the node list cannot apply.
        """
        refused: List[str] = []
        owner = type(self)
        if (owner.run_generation is not EMRHyperNEAT.run_generation
                or owner.run_generation_verbose
                is not EMRHyperNEAT.run_generation_verbose):
            refused.append(f"{owner.__name__}, which replaces the generation that runs the node "
                           "list")
        depth = (node_list.leaf_depth(self.max_depth)
                 if isinstance(self.max_depth, numbers.Integral) else None)
        if (depth is None or isinstance(self.max_depth, bool)
                or not 1 <= depth <= node_list.MAX_DEPTH):
            refused.append(f"max_depth {self.max_depth!r} (the quadtree depth of the leaves: an "
                           f"integer from 1 to {node_list.MAX_DEPTH}; from depth 6 HMR's own build "
                           f"dropped every hidden connection, and a genome's search is "
                           f"4^(2 max_depth) leaf pairs)")
        if (isinstance(self.max_weight, bool) or not isinstance(self.max_weight, numbers.Real)
                or not self.max_weight > 0):
            refused.append(f"max_weight {self.max_weight!r} (a positive number)")
        for key in ('variance_threshold', 'division_threshold', 'initial_depth', 'variance_rule'):
            if key in hmr_config:
                refused.append(f"emr_hyperneat.{key} (the node list has no variance step: every "
                               f"leaf is a candidate, as in HMR's model, whose active-leaf mask "
                               f"was all true)")
        if 'weight_threshold' in hmr_config:
            refused.append("emr_hyperneat.weight_threshold (the grid's pruning; the node list "
                           "sparsifies at HMR's fixed 0.2 and keeps a connection by the band test)")
        for key in self.NODE_LIST_REFUSED_RECURRENCE_KEYS:
            if key in recurrence_section:
                refused.append(f"recurrence.{key} (the grid model's)")
        if recurrence_section.get('hh_cache_enabled', False):
            refused.append("recurrence.hh_cache_enabled (the node list discovers every genome's "
                           "network every generation)")
        if self._settling_rule != 'emr':
            refused.append(f"recurrence.settling_rule {self._settling_rule!r} (the node list "
                           f"settles by HMR's node-list update)")
        config = self.extended_config
        hidden_to_hidden = config is not None and bool(config.allow_hidden_to_hidden)
        rounds = int(config.iteration_level) if hidden_to_hidden else 0
        if hidden_to_hidden and rounds < 1:
            refused.append("recurrence.iteration_level 0 with hidden-to-hidden connections (it is "
                           "the node list's rounds of phase-2 discovery: 1 or more, or switch "
                           "hidden-to-hidden connections off)")
        if not hidden_to_hidden and recurrence_section.get('iteration_level', 0) != 0:
            refused.append(f"recurrence.iteration_level {recurrence_section['iteration_level']!r} "
                           f"without hidden-to-hidden connections (no phase 2 runs)")
        steps = self.activate_time
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 2:
            refused.append(f"activate_time {steps!r} (at least 2: the node list reads its outputs "
                           f"one step behind the hidden nodes, so 1 step scores every genome as "
                           f"the constant 0.5 network)")
        if self.dynamic_functions_mode != 'disabled':
            refused.append(f"dynamic_functions.mode {self.dynamic_functions_mode!r}")
        if self.df_hidden_activation != 'tanh' or self.df_output_activation != 'sigmoid':
            refused.append(f"hidden_activation {self.df_hidden_activation!r} and "
                           f"output_activation {self.df_output_activation!r} (the node list's "
                           f"nodes are tanh, its outputs sigmoid)")
        if self.agg_mode != 'disabled' or self.agg_palette is not None:
            refused.append(f"aggregation.mode {self.agg_mode!r} or an aggregation palette")
        if self.neuromod_config is not None and self.neuromod_config.mode != 'disabled':
            refused.append(f"neuromodulation.mode {self.neuromod_config.mode!r}")
        if self.multitask_config is not None and getattr(self.multitask_config, 'enabled', False):
            refused.append("multi-task evaluation")
        for name, on in (('LEO', self.use_leo), ('geometry seeding', self.geometry_seeding_enabled),
                         ('locality_radius', self.locality_radius is not None),
                         ('swarm mode', self.swarm_mode)):
            if on:
                refused.append(name)
        cppn_genome = (algo_params.get('cppn') or {}).get('genome') or {}
        if (self.num_cppn_outputs != 1 or cppn_genome.get('num_outputs', 1) != 1
                or cppn_genome.get('num_inputs', 5) != 5):
            refused.append("a CPPN other than 5 inputs and 1 output (the node list reads the "
                           "weight output alone)")
        execution = hmr_config.get('execution') or {}
        if execution.get('generation_loop', 'python') != 'python':
            # HMR's GPU loop compiles the plain feed-forward generation only
            refused.append(
                f"execution.generation_loop {execution['generation_loop']!r} (the node list "
                f"runs the Python loop)"
            )
        if (hmr_config.get('enable_streaming') or set(execution) - {'mode', 'generation_loop'}
                or execution.get('mode', 'auto') not in ('auto', 'standard')):
            refused.append("an execution block other than mode standard or auto, or "
                           "enable_streaming (the node list has one execution path)")
        if self.strategy not in (MultiGPUStrategy.SINGLE_GPU,
                                 MultiGPUStrategy.SINGLE_GPU_PYTHON_LOOP):
            refused.append(f"strategy {self.strategy.name} (the node list runs on one device)")
        inputs, outputs = self.substrate_input_coords, self.substrate_output_coords
        if not inputs or not outputs:
            refused.append("a substrate without input or output coordinates")
        elif depth is not None and 1 <= depth <= node_list.MAX_DEPTH:
            clash = node_list.coincident_coordinates(depth, inputs, outputs)
            if clash:
                refused.append(f"input or output coordinates {list(clash)} on a leaf or on each "
                               f"other (the nodes would merge)")
        chunk = self.population_chunk_size
        if isinstance(chunk, bool) or not isinstance(chunk, numbers.Integral) or chunk < -1:
            refused.append(f"population_chunk_size {chunk!r} (-1, 0 or a positive count of "
                           f"genomes per call)")
        if refused:
            raise ValueError("recurrence.network_model 'node_list' cannot apply: "
                             + "; ".join(refused))
        classes = ((bool(config.allow_backward), bool(config.allow_lateral),
                    bool(config.allow_self_loops))
                   if hidden_to_hidden else (False, False, False))
        self._node_list_settings = node_list.NodeListSettings(
            depth=depth, rounds=rounds, classes=classes,
            band_threshold=float(self.band_threshold), max_weight=float(self.max_weight),
            activate_time=int(steps), population_chunk=int(chunk))

    def initialize(self, config: Any, problem: Any, seed: int = 42) -> Any:
        """Initialize with per-device JIT functions for multi-GPU support.

        This overrides the parent initialize() to add per-device JIT function
        creation. This fixes device placement errors when using ThreadPoolExecutor
        for device-parallel h->h processing - JAX JIT functions are compiled to a
        specific device and ignore thread-local device contexts.

        Args:
            config: Algorithm configuration
            problem: Problem instance
            seed: Random seed

        Returns:
            Initialized algorithm state
        """
        # A run's fixed random draws are its own: an object reused for the next seed draws them
        # again (scope 'shared' draws the same in every run, so for it this changes nothing).
        self._random_fixed_indices = None
        self._random_fixed_agg_indices = None
        # The generation count is the run's too: random_generation keys on it, so a reused object
        # must not continue it.
        self._current_generation = 0
        # The island model derives every island's seed from the run's (island_seed)
        self._run_seed = int(seed)
        # A compiled generation loop holds the last run's data shapes and compiled functions
        self._generation_loop_cache = {}
        # And so is the h->h cache: only create_config built it, so an object reused through
        # initialize alone served the new run the last run's connections until its generation count
        # passed the last refresh by hh_refresh_interval (or the mask changed)
        if self.extended_config is not None:
            self._hh_cache = HHCacheManager(self.extended_config)

        # Call parent initialize first
        state = super().initialize(config, problem, seed)

        # Palette evolution's state is the run's: HMR kept it across initialize, so a reused object
        # started from the previous run's best fitness and palette. The key is HMR's (seed + 12345).
        evolution = getattr(self, 'df_palette_evolution', None)
        if evolution is not None:
            self._palette_mask = evolution['initial_mask'].copy()
            self._palette_rng_key = jax.random.PRNGKey(seed + 12345)
            self._palette_best = None
            self._palette_stagnation = 0
            self.df_palette = jnp.asarray(np.flatnonzero(self._palette_mask), dtype=jnp.int32)
            self._palette_owned = np.asarray(self.df_palette)

        # Create per-device JIT functions for multi-GPU support
        # This fixes device placement errors when using ThreadPoolExecutor
        devices = jax.devices()
        self._jitted_cppn_forward_per_device = {}

        if len(devices) > 1:
            print(f"[Multi-GPU] Creating per-device JIT functions for {len(devices)} devices")
            for device in devices:
                # Use device= parameter to pin JIT function to specific device.
                # jax.default_device() context only affects where arrays are created,
                # it does not pin JIT functions to that device. Without device=, the
                # JIT function can be reused across devices causing device placement errors.
                self._jitted_cppn_forward_per_device[device] = jax.jit(
                    self.neat_algo.genome.forward, static_argnums=(0,), device=device
                )

        return state

    def refresh_problem_cache(self, problem: Any) -> None:
        """Refresh cached problem data for MVG (Modularly Varying Goals) support.

        This method re-reads the problem data and updates the cached inputs and
        targets. Use this when the problem's targets change during evolution
        (e.g., Kashtan MVG where goals alternate every N generations).

        Args:
            problem: Problem instance with get_data() method.

        Example:
            # In evolution loop with MVG problem:
            for gen in range(max_generations):
                new_goal = problem.set_generation(gen)
                if problem.needs_cache_refresh():
                    algo.refresh_problem_cache(problem)
                    problem.mark_cache_refreshed()
                state, metrics = algo.run_generation(state, problem)
        """
        # Get fresh data from problem
        if hasattr(problem, 'get_data'):
            data = problem.get_data()
            self._cached_inputs = jnp.array([d[0] for d in data], dtype=jnp.float32)
            self._cached_targets = jnp.array([d[1] for d in data], dtype=jnp.float32)
        elif hasattr(problem, 'inputs') and hasattr(problem, 'targets'):
            self._cached_inputs = jnp.array(problem.inputs, dtype=jnp.float32)
            self._cached_targets = jnp.array(problem.targets, dtype=jnp.float32)
        else:
            raise ValueError(
                "Problem must have get_data() method or inputs/targets attributes"
            )

        # Mark problem cache as refreshed if it supports this interface
        if hasattr(problem, 'mark_cache_refreshed'):
            problem.mark_cache_refreshed()

    def run_generation_vmapped_multires(
        self, state: Any, problem: Any, skip_metrics: bool = False
    ) -> Tuple[Any, AlgorithmMetrics]:
        """HMR's name, which its base class bound to its Python loop: this class's generation
        (run_generation). The base class's binding (run_generation_vmapped_multires =
        run_generation_verbose) named the base class's own loop, which skips this class's
        evaluation."""
        return self.run_generation(state, problem, skip_metrics)

    run_generation_vmapped_multiresA = run_generation_vmapped_multires


# The names of the base class's own runners, removed with its generation loops: each built and
# scored networks with the base class's pipeline instead of EMRHyperNEAT's evaluation and routes,
# and run_until_threshold_data_parallel also weighted a padded dataset's last sample several times.
# EMRHyperNEAT refuses them, as the base class refuses run_until_threshold_with_fixed_hh.
_BASE_ONLY_RUNNERS = (
    'run_until_threshold_multi_gpu', 'run_until_threshold_dual_gpu_eval',
    'run_until_threshold_population_parallel', 'run_until_threshold_periodic_sync',
    'run_until_threshold_overlapped', 'run_until_threshold_periodic_overlapped',
    'run_until_threshold_gpu_resident_multi_gpu', 'run_until_threshold_gpu_resident_shardmap',
    'run_until_threshold_batched_multi_gpu', 'run_until_threshold_data_parallel',
    'run_until_threshold_streaming',
)


def _refused_base_runner(name: str) -> Callable[..., Dict[str, Any]]:
    """The EMRHyperNEAT method that refuses `name`, a runner the base class had."""
    def runner(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{name} is refused on EMRHyperNEAT: it was the base class's own runner, which "
            f"skipped EMRHyperNEAT's evaluation and routes. Call run_until_threshold; "
            f"multi_gpu_strategy and execution.mode choose the route.")
    runner.__name__ = name
    runner.__qualname__ = f'EMRHyperNEAT.{name}'
    runner.__doc__ = ("Refused: the base class's own runner (_BASE_ONLY_RUNNERS); call "
                      "run_until_threshold.\n\nRaises:\n    NotImplementedError: Always.")
    return runner


for _name in _BASE_ONLY_RUNNERS:
    setattr(EMRHyperNEAT, _name, _refused_base_runner(_name))
del _name


# Class Aliases

# Alias for "Full" variant (with neuromodulation support)
EMRHyperNEATFull = EMRHyperNEAT

# Exports

__all__ = [
    # Main class (both names)
    'EMRHyperNEAT',
    'EMRHyperNEATFull',

    # Neuromodulation dataclasses and config
    'NeuromodulationConfig',
    'MultiTaskConfig',
    'NEUROMODULATION_PRESETS',
    'NT_TASK_PRESETS',
    'derive_receptor_from_weight',

    # Dataclasses
    'SparseHiddenConnections',
    'EMRConfig',
    'EMRRecurrenceMetrics',

    # Presets
    'RECURRENCE_PRESETS',
    'get_recurrence_preset',

    # Forward pass
    'ForwardPassMode',
    'forward_hybrid_vmapped',
    'eval_single_network_hybrid',

    # Multi-hop
    'compute_multi_hop_connections_matrix_power',
    'compute_multi_hop_connections_fori_loop',

    # Discovery
    'discover_sparse_hh_vectorized_multi_hop',

    # Constraint filtering
    'get_connection_constraint_mask_batched',

    # Cache
    'HHCacheManager',

    # Multi-GPU (re-export from parent)
    'MultiGPUStrategy',

    # The split routes' device grid and the island model
    'DeviceSplit',
    'make_device_split',
    'IslandPlan',
    'island_seed',
]


# A name set on this module from outside, as a test's monkeypatch or a harness replacing a function does, is also
# set in every emr_hyperneat.engine module that holds the same object under that name. The engine's code moved
# to those modules, and each reads its globals from its own module, so a patch reaches it as it did when the
# code lived here; setting the name back restores it everywhere. Assignments in this file itself are unaffected.
_MISSING = object()


class _EngineFacade(types.ModuleType):
    """The type of this module: setting a name also sets it in the engine modules that hold the same object."""

    def __setattr__(self, name: str, value: Any) -> None:
        old = self.__dict__.get(name, _MISSING)
        if old is not _MISSING:
            for module_name, module in tuple(sys.modules.items()):
                if (module_name.startswith('emr_hyperneat.engine.') and module is not None
                        and module.__dict__.get(name, _MISSING) is old):
                    module.__dict__[name] = value
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EngineFacade
