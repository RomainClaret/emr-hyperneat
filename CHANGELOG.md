# Changelog

Notable changes to the released `emr_hyperneat` package. The version tracks
`emr_hyperneat/_version.py`.

## 0.2.0

Results of recurrent runs change: one settling rule, a multi-hop expansion within each genome's own
connections, and new defaults.
[docs/configuration.md](docs/configuration.md#reproducing-a-010-recurrent-run) lists the settings
that reproduce a 0.1.0 recurrent run. Runs on the multi-device routes of `run_until_threshold`
compute what a single-device run computes, and the HMR engine's split routes and island model are
routes of it. The HMR engine's features are settings of `EMRHyperNEAT`, off by default. The frozen
HMR module is no longer shipped: the paper scripts that ran it are EMR versions, and their published
runs reproduce on 0.1.0
([docs/reproducing-experiments.md](docs/reproducing-experiments.md#choosing-a-release)). Keys and
values that 0.1.0 ignored raise. On a CPU, the per-node paper's EMR runners give the same results as
with 0.1.0; the neuromodulation paper's runner departs from 0.1.0, because 0.2.0 applies the
`neat_species` block it passes.

### Added

- **Substrate API** (`emr_hyperneat.substrate`), the functions the generation loop builds its
  substrate with: `population_substrates`, `substrate_queries`, `build_substrate` and
  `substrate_forward` for the feed-forward substrate; `hh_query_rows`, `build_recurrent_substrate`
  and `query_recurrent_substrate`, which build the loop's hidden-to-hidden connections bit for bit;
  and `return_states=True` for the settling states. `substrate_config()` refuses what it cannot
  describe.
- **The hidden-to-hidden cap's policy** (`recurrence.sparse_cap_policy`): what a genome whose
  discovered set exceeds `max_sparse_conns` meets. `refuse`, the default, raises
  `SparseCapExceeded` (a `ValueError`, under Refused, with each genome's count before the cap and
  the indices over it); `truncate` keeps its first pairs, as 0.1.0
  did, with a `SparseCapWarning`; `grow` raises the cap to hold the largest set, up to
  `recurrence.max_sparse_conns_ceiling`, and discovers the generation again. `set_sparse_cap`
  changes the cap, the policy or the ceiling between generations; `run_until_threshold`'s result
  carries `sparse_cap`, the cap the run ended with and how it got there; the substrate API follows
  `SubstrateConfig.sparse_cap_policy`. `SparseCapExceeded`, `SparseCapWarning` and
  `SPARSE_CAP_POLICIES` are exported.
- **Per-genome h->h discovery** (`recurrence.hh_discovery_mode`): each genome is queried over its
  own active positions in static buckets, and gets the union query's connections and fitness, bit
  for bit on a CPU.
- **Bucketed compaction** (`execution.compaction_bucket`): the recurrent evaluation pads its active
  set to a bucketed length, so a length compiles once per run. Results equal `exact` bit for bit on
  an arm64 CPU; on x86 a padded sum can round differently, moving a fitness by a float32 ulp or two.
- **Records**: `run_until_threshold` returns its `route`, `devices` and `execution_mode`; per-genome
  counts of the last h->h discovery in the module-level `emrhyperneat._last_discovery_stats`,
  including each genome's count before the `max_sparse_conns` cap; `nan_fitness_count`;
  `hh_max_uncapped` (the largest set), `hh_truncated_genomes` (0 unless `sparse_cap_policy:
  truncate`) and `hh_max_sparse_conns` (the cap the discovery used) in the generation's metrics;
  the run's `sparse_cap` record (below). The generation's
  metrics report the evaluated population's CPPN species (count, sizes, best fitness of each), its
  topology (hidden nodes and connections, mean and spread), `num_levels` and `variance_threshold`
  on the standard, multi-task and depth-offload paths (0.1.0 reported no species, or one holding the
  whole population, and no topology).
- **`index_decode`** (`dynamic_functions.index_decode` and `aggregation.index_decode`; in the
  substrate API, `SubstrateConfig.activation_index_decode` and `aggregation_index_decode`, and
  `decode=` in `cppn_output_function_indices` and `continuous_to_index`): `uniform` bins a CPPN
  channel or a weight-interpretation signal as it is, every bin equally wide. The default, `tanh`,
  bins `tanh` of a value already in `[-1, 1]`, as 0.1.0 does, so a palette of 9 or more functions
  never reaches its end entries: with all 18 activations, positions 2 to 15 only, and no node is
  assigned `tanh`, `sigmoid`, `band_pass` or `integrate`. The default keeps 0.1.0's results.
- **Paper releases**: each paper names the release its code was published with
  (`papers/releases.json`; 0.1.0, tag `v0.1.0`, for the four papers), and `scripts/replicate.py`
  runs a paper's commands on that release in its own checkout and environment, or on another one
  with `--release`.
- **The HMR engine's features, as settings of `EMRHyperNEAT`**, each off unless configured. HMR's
  defects are fixed, not reproduced, and the tests check many of them against values recorded from
  the HMR engine (Tests and CI).
  - Recurrence: `recurrence.settling_rule: hmr`, HMR's settling rule, also in the substrate API's
    recurrent forward, with HMR FULL's additive rule under static gating, context gating or true
    neuromodulation; `dynamic_functions.hh_activation_mode: initial_only` under it.
    `recurrence.network_model: node_list`, HMR's node-list recurrent model, at depths 1 to 5.
    `recurrence.hh_cache_trigger` (`time_or_mask`, the default, or `time`).
  - Per-node functions: `dynamic_functions` mode `modular`, HMR's activation, sparsity and scaling
    layers, and the `modular` aggregation method; HMR's weight interpretations `magnitude`,
    `combined`, `hierarchical_sparsity`, `sparsity_threshold` and `stp_inspired`, with the
    `sparsity` blocks they read; HMR's reads (`cppn_read: input_mean`, `weight_source: raw`); an
    aggregation palette (`aggregation.palette`, or `agg_palette` set by a runner); built-in palette
    evolution (`dynamic_functions.palette_evolution`, under mode `cppn_output`); HMR's live palette
    strategy (`dynamic_functions.live_strategy`, `set_palette_strategy`).
  - Neuromodulation: HMR's mode names (its Option A and Option B names, `xdg_style`,
    `modulatory_only`, and `custom` with level flags); Option B, receptor densities from dedicated
    CPPN outputs (`receptor_from_weight: false`; 0.1.0 raised `NotImplementedError` for it at the
    first generation); gates and modulatory neurons from dedicated CPPN outputs
    (`gate_source: cppn`, HMR's `gate_from_cppn`, `modulatory_source: cppn`) and the
    modulatory-neuron options (selection, count rule, signal, node weights, scope, CPPN read, CPPN
    outputs, matched gain, `mod_neuron_count`); `node_rule: anti`, HMR ANTI's rule.
  - Multi-task: HMR's call form, `run_generation_multitask(state, MultiTaskConfig(tasks={name:
    (inputs, targets)}, ...))`, returning `MultiTaskMetrics` by task name; per-genome orthogonality
    and specialization bonuses (`orthogonality_bonus_weight`, `specialization_bonus_weight`,
    `orthogonality_metric`); branch gating (`branch_gating_mode`); the modulation penalty
    (`modulation_penalty`). 0.1.0 accepted the last two at any value and applied neither.
  - LEO (link-expression output): the `leo` block (`compound`, `replacement`, and `hmr_compound`,
    HMR's gate), on every path that builds weights in the standard and streaming modes.
  - `emr_hyperneat.weight_threshold`, the input-to-hidden and hidden-to-output pruning threshold:
    0.1 by default, which 0.1.0 always used.
  - A configured `output_activation` on the plain path (no per-node function, aggregation or
    neuromodulation) runs as mode `global` with a `tanh` hidden activation, which equals the plain
    evaluation bit for bit; 0.1.0 ignored it there and applied a sigmoid. With `aggregation.mode:
    modular` and method `none`, which aggregate nothing, it raises.
  - HMR's names: `NUM_ACTIVATIONS`; `NT_PRESETS` and the mnist and iris vectors;
    `get_cppn_num_outputs` and `get_cppn_receptor_base_idx`; `cppn_num_outputs`, `aggregation_mode`,
    `_hidden_act_func` and `_output_act_func`; `run_generation_vmapped_multires` and
    `run_generation_vmapped_multiresA`; `configure_offloading(skip_dead_levels=)`;
    `run_until_threshold`'s keys `converged`, `solved`, `final_fitness` and `sparse_hh_stats`; UNI's
    `get_recurrence_metrics`, `sparse_hh_count`, `hh_cache_status`, `_use_sparse_hh` and
    `_use_full_recurrent`; the four single-network true-neuromodulation evaluators; HMR's three
    context-gated multi-task methods, which raise.
  - Continual learning: `get_best_genome_with_fitness`, `inject_genome_into_population`,
    `freeze_topology`, `unfreeze_topology`, `is_topology_frozen`, `evaluate_genome_on_problem` and
    `create_state_for_new_problem`.
  - `execution.generation_loop` (`"python"`, the default, or `"while_loop"`): the HMR engine's GPU
    loop over the plain feed-forward generation, compiled once inside `jax.lax.while_loop`. Speed
    only, within one float32 ulp of fitness; it refuses every setting outside that generation. See
    [the generation loop](docs/configuration.md#the-generation-loop).
  - **The split routes** (`multi_gpu_strategy: population_pmap`, `position_sharding` or `hybrid`):
    `single_gpu`'s generation with its one-output CPPN query and its evaluation spread over a grid
    of devices, which the `emr_hyperneat.parallel` block sets (`population_devices`,
    `position_devices`). The query runs on each device in chunks of at most 100 genomes, split by
    genomes, by grid positions or both, and the evaluation is split by genomes; the rest of the
    generation, hidden-to-hidden discovery at any iteration level included, runs on the first
    device. Every setting of the standard generation runs, and on CPU devices a run equals
    `single_gpu`'s bit for bit (on GPUs, to float32 rounding). The HMR engine's names map onto them:
    `population_parallel` is `population_pmap`; `position_pmap`, `shard_map`, `cppn_chunked` and
    `position_sharding_chunked` are `position_sharding`. See
    [`parallel`](docs/configuration.md#parallel-the-split-routes-device-grid). In 0.1.0 these names
    ran on one device (Changed gives the two chunked ones): the configuration key warned and ran
    `single_gpu`, and the constructor ran the one-device loop under the name.
  - **The island model** (`multi_gpu_strategy: island_model`, and the HMR engine's
    `island_model_v2`): islands of the population evolve apart in lockstep, each a `single_gpu` run
    on its device at its own population and seed, as the `emr_hyperneat.islands` block sets
    (`count`, `populations`, `devices`, `migration_interval`, `migrants`, `topology`). Every
    `migration_interval` generations each island's best `migrants` genomes go, with their fitness,
    to the next island (`ring`), to every other (`star`) or to one drawn from the run's seed
    (`random`), and replace the receiver's worst genomes in its next selection step. Island i's seed
    is drawn from the run's seed and i (`island_seed`), where the HMR engine's `seed + i` made
    island 1 of seed 42 island 0 of seed 43. See
    [`islands`](docs/configuration.md#islands-the-island-model). 0.1.0 ran both names on one device,
    as the split routes' names.

### Changed defaults (the setting in parentheses restores 0.1.0)

- **The h->h connection cache is off** (`recurrence.hh_cache_enabled: true`). Its connections are
  indexed by population slot, so between refreshes a genome could be evaluated with connections
  discovered for another genome.
- **Iteration level 1** for the recurrent presets, `EMRConfig` and explicit settings
  (`iteration_level: 2`).
- **`multi_hop_second_factor: masked`** (`raw`) and **`hh_discovery_mode: per_genome`** (`union`).
- **`execution.compaction_bucket: auto`** (`exact`: 0.1.0's shapes, which an x86 CPU or a GPU needs
  to match 0.1.0 bit for bit).
- **`random_scope: genome`** for the random function modes of `dynamic_functions` and `aggregation`:
  each population slot draws its own functions from the run's key (`random_scope: shared`). In 0.1.0
  one assignment was drawn from a fixed key for every genome and every run, so a random-control arm
  compared the evolved assignment with a single random one. A reused algorithm object also draws
  again for its next run.
- **`neuromodulation.node_rule: paper`**, the published neuromodulated node, on every path
  (`node_rule: hmr_full`). In 0.1.0 the single-task path computed HMR FULL's rule (an unclipped
  gain, no bias, the gate `sigmoid(5 m)`) and the multi-task path the published one (a gain clipped
  to [0.1, 5], an additive bias, the gate `sigmoid(m)`). The output inversion acts only when
  `use_output_inversion` is set and the vector has four channels or more: 0.1.0's multi-task path
  read the fourth channel whatever the count, so with two or three channels a modulation channel
  also inverted the outputs, and it ignored the flag.
- **`hh_activation_mode: every_iteration`**. 0.1.0's default, `initial_only`, is refused outside
  `recurrence.settling_rule: hmr`; `tanh` settles as it did when no input aggregation is configured.

### Changed

- **Mode `modular`** computes HMR's three layers; 0.1.0 warned that it was not implemented and
  evaluated fixed `tanh`. With `aggregation.mode: modular` its aggregation method applies; 0.1.0 ran
  `sum`.
- **A runner's `agg_palette` takes effect** (0.1.0 read no aggregation palette), and the palettes a
  runner sets between generations (`df_palette`, `agg_palette`) are checked every generation: an
  entry outside its function list raises. 0.1.0 failed with an `IndexError` at evaluation on an
  activation entry beyond the list and silently wrapped a negative one.
- **`cppn_chunked` and `position_sharding_chunked` run the position split** (`position_sharding`), a
  CPPN query split by grid position, as the HMR engine ran them. 0.1.0's configuration key ran
  `cppn_chunked` as `streaming`, one device querying one level at a time, and
  `position_sharding_chunked` as `full_pipeline_parallel`, the data-parallel pipeline. A run under
  either name computes what `single_gpu` computes, bit for bit on a CPU; name `streaming` or
  `full_pipeline_parallel` for the route 0.1.0's configuration key took.
- **The engine's layout**: its code is in the package `emr_hyperneat.engine`, by concern, and
  `emr_hyperneat.emrhyperneat` assembles `EMRHyperNEAT` from its mixins and imports every name back.
  Every name keeps its import path there and is the same object (a registry or a cache is the
  engine's own), and a name set on `emr_hyperneat.emrhyperneat`, as a test's monkeypatch does, also
  reaches the engine module that reads it. The classes that moved report their new module in
  `__module__`.

### Removed

- `emr_hyperneat/_hmr_frozen/`, the frozen HMR engine (8 modules, 53,215 lines), and its tests
  (`test_hmr_frozen.py`). No module, test or paper script imports it; release 0.1.0 ships it, and
  `scripts/replicate.py` runs the published originals there.
- The code in `_compat/utils/jax_smart_config.py` that added `--xla_gpu_deterministic_ops=true` to
  `XLA_FLAGS`. Nothing in 0.1.0 reached it.
- `forward_unified_extended`, `get_connection_constraint_mask` and `PositionShardingConfig`, which
  0.1.0 exported from `emr_hyperneat.emrhyperneat`. Nothing in 0.1.0 called the first two
  (`get_connection_constraint_mask_batched` stays); `PositionShardingConfig` configured the base
  class's own position sharding, which `EMRHyperNEAT` never ran.
- Code that nothing reached: in the vendored `_compat` shim, most methods of the TensorNEAT adapter
  and of its base class, the parameter translator's web-UI, duplicate and validation helpers, the
  config manager's unused loaders and its legacy mapping table, unused precision and metrics helpers,
  and `_compat/schemas/experiment_schema.py`; elsewhere, unused imports, local variables and instance
  attributes. No public name and no result changes.

### Fixed

- **`get_best_genome(state)` returns the champion** of the last evaluated generation, `(nodes, conns)`,
  as `get_best_genome_with_fitness` does, and raises before the first generation. 0.1.0 returned the
  first element of what TensorNEAT's `ask` returns: the whole population's node array.
- **One settling rule for every recurrent evaluation**, `h = f(drive + h @ Whh)` from zero, with the
  node's activation, input aggregation and neuromodulation. In 0.1.0 the settling step applied no
  input aggregation and, by default, no configured activation; its neuromodulated evaluators used a
  residual update, and `eval_only_parallel` an older rule.
- **`substrate.activate_time` applies with recurrence** (in 0.1.0 the recurrence settings replaced
  it); two blocks that set it must agree, and `set_recurrence_config` resolves the steps, level and
  forward mode the same way.
- **The multi-hop expansion composes the genome's own connections**, within its class. In 0.1.0 it
  composed over the population's active positions, so a composite connection could end on positions
  the genome did not have, fall outside its class, and depend on the other genomes.
- **The `full_pipeline_parallel` pipelines apply the connection class**; in 0.1.0 they evaluated
  every preset as full recurrence without self loops.
- **Iteration level 0 discovers the direct connections on every path**, as level 1 does. In 0.1.0
  depth offload and the `full_pipeline_parallel` pipelines evaluated level 0 feed-forward; depth
  offload also required `enabled`, and the pipelines recorded `effective_iteration_level` 0 at every
  level. Depth offload now settles hidden-to-hidden connections whenever the configuration allows
  them.
- **`fitness_mode` applies on the recurrent and neuromodulated paths** (in 0.1.0 these paths used
  `mse`), and **a NaN fitness becomes `-inf` before selection on every route**; in 0.1.0 only the
  `full_pipeline_parallel` pipelines and depth offload converted it.
- **Multi-task**: `multitask.fitness_mode` is read and `accuracy_confident` is computed (in 0.1.0
  both evaluated `mse`), and the generation counter advances once per generation (twice in 0.1.0).
- **`neat_mutation`, `neat_species` and `neat_selection` reach TensorNEAT**, over the defaults 0.1.0
  always used; an unknown setting raises. In 0.1.0 a configuration's values for them were not
  applied (drop the blocks to reproduce such a run).
- **`weight_interpretation` with a `palette` gives every node a palette function**, as `cppn_output`
  and the random modes do. In 0.1.0 the decoded palette position was not mapped to the function, so
  a node whose position was not itself a palette member output 0.
- **`random_fixed` runs with hidden-to-hidden connections**, in `dynamic_functions` and in
  `aggregation`: the fixed draw covers the whole position axis and each generation takes it at the
  positions it evaluates, so a grid position keeps its function from one generation to the next. In
  0.1.0 the draw had the length of generation 0's active positions and was reused whole, so the run
  failed with "Incompatible shapes for broadcasting" at the first generation whose active count
  differed.
- **A strategy name passed to the constructor takes the route it has as `multi_gpu_strategy`**
  (`STRATEGY_NAME_MAP`); `EMRHyperNEAT()` still runs `single_gpu`. In 0.1.0 only `streaming`,
  `eval_only_parallel` and `population_parallel_process` kept their multi-device route through the
  constructor, and every other name ran the one-device loop under the requested name.
- **A neuromodulation preset runs its level**: it sets the mode (`xdg_style` is `context_gating`),
  and `use_self_connection_query` reaches it. In 0.1.0 a preset left the mode `disabled`, so a run
  configured by a preset alone had no modulation.
- **Modulatory neurons are each genome's own** (neuromodulation mode `modulatory_neurons`):
  `mod_neuron_ratio` (0.1) of a genome's own active positions, at least one, those of highest
  input-weight variance, and its forward pass averages their activity alone. In 0.1.0 every genome
  took the same positions, that fraction of every grid position (34 at depth 3, 546 at depth 5) at
  the top of the population's mean variance, so a genome's network depended on the rest of the
  population and most of its modulatory positions were positions it did not have (in a
  three-generation XOR run at those depths, a genome had a median of 4 active positions). No paper
  of the release uses the mode.
- **The multi-task settings reach the evaluation** (`softmin_temperature`, `modulation_mode`,
  `modulation_strength_override`, `two_module_mode` and the generalist bonus). In 0.1.0 the parser
  dropped them, so they acted only when set on a `MultiTaskConfig` object, and two-module mode read
  a cached grid that nothing set and always fell back to one module.
- **`fitness_mode: acc_mse` and `accuracy_confident` on the single-task paths** compute what the
  multi-task evaluators compute (accuracy plus 0.01 of 1 - MSE; accuracy plus 0.1 of the outputs'
  mean distance from 0.5). In 0.1.0 both names computed MSE there.
- **`substrate.hidden_activation` and `output_activation` are the defaults of the
  `dynamic_functions` activations**, as in HMR's dynamic-functions variant; 0.1.0 read neither key.
- **With `weight_interpretation` aggregation the hidden-to-hidden side follows the
  interpretation**; 0.1.0 ran `magnitude_bio` there for `sign_uniformity` and `fan_in`.
- **`variance_rule: legacy` applies on every execution mode.** In 0.1.0 streaming and depth_offload
  computed the canonical level-0 variance under `legacy`. A configured mode warned, but a mode
  chosen at run time (`auto`'s memory-aware escalation on a GPU) did not, so such a run used the
  canonical rule silently. The configuration warning is gone.
- **Depth offload reports the mean, min and std fitness over the finite values** and counts NaN
  fitness in `nan_fitness_count`, as the standard path does; in 0.1.0 a NaN fitness, told as
  `-inf`, made its mean fitness `-inf`.
- **`evaluate_on_data` scores the evolved champion** (in 0.1.0 it scored a dense `tanh` network
  rebuilt from the champion).
- **An aggregation-only `cppn_output` CPPN reads its second output**; 0.1.0 read the weight output
  there, which `cppn_output_indices: {aggregation: 0}` reproduces.
- **The multi-GPU worker's cache path runs**; in 0.1.0 it built `EMRConfig` with a field that class
  does not have, which raised an error.
- **Nothing compiles again every generation.** 0.1.0 compiled again, every generation:
  - the recurrent evaluation's settling loop, with every node function: the named ones, per-node
    activations and the neuromodulated nodes, now rules made once per setting whose terms the
    settling step carries;
  - multi-task evaluation and `eval_chunk_size` chunking;
  - the self-connection query;
  - the multi-hop expansion of the union discovery, which built its `jit(vmap)` on every call.

  Each recompiled loop kept its compiled code for the rest of the run. On a Linux CPU (JAX 0.6.1), a
  recurrent XOR run at depth 3 and population 100 grew by about 2 MiB per generation, to 3,612 MiB
  after 1,000 generations; it now holds near 1,400 MiB after the first 150. The values are the same,
  bit for bit on a CPU.
- **The multi-device routes of `run_until_threshold`:**
  - **`eval_only_parallel` padding.** A sample count that is not a multiple of the device count was
    padded with zero rows, each adding 0.25 per output to every genome's error, so every fitness was
    0.25 times the padding over the sample count too low (XOR on three devices: 0.125), and a target
    fitness above that could not be reached. The padding rows now have zero inputs and 0.5 targets:
    the route computes tanh hidden nodes, a sigmoid output and no bias term, so a zero row outputs
    exactly 0.5 for every genome and adds no error. The ranking, and so selection, was the same in
    0.1.0; the reported fitness and the stop test were not.
  - **`full_pipeline_parallel` padding.** The dataset was padded with copies of its last sample and
    the shards' fitnesses averaged, so the last sample counted once per copy, weighted by each
    genome's own error on it: the ranking, and so selection, could change (33 samples on two
    devices: best fitness 0.7651 against 0.7793 on `single_gpu`). The padding rows now add no error,
    as above, and the shards are combined over the real samples.
  - **The random keys.** `full_pipeline_parallel` split the key before each generation's ask by the
    base class's own setting, which defaults on, while a single-device run splits it only when
    `extra_randkey_split` is set; `eval_only_parallel` and `population_parallel_process` never split
    it. Every route now reads `extra_randkey_split`.
  - **`population_parallel_process` with hidden-to-hidden connections.** The worker's own h->h loop
    indexed the sample axis and failed on every shard, which kept a fitness of 0 for every genome.
    The worker now scores h->h connections with the kernel `eval_only_parallel` uses, and a shard
    that fails raises instead of scoring 0.
  - **`population_parallel_process` device placement.** Each worker set `CUDA_VISIBLE_DEVICES` after
    the package import had initialized JAX, so every worker saw every GPU and ran on the first. The
    route now starts one single-worker process pool per device, at the first generation, with that
    device's `CUDA_VISIBLE_DEVICES` (the device's entry in the parent's own list when one is set)
    and `XLA_PYTHON_CLIENT_PREALLOCATE=false` (unless an `XLA_PYTHON_CLIENT_*` memory variable is
    set) in the worker's environment from its start, and keeps the pools for the whole run. A worker
    that sees more than one GPU raises.
  - **`population_parallel_process` numeric settings.** A spawned worker reads only its environment
    when its JAX starts, so a matmul precision or an x64 mode set with `jax.config.update` (as a
    test conftest or a script does) never reached it: 0.1.0's workers ran at JAX's default
    precision, TF32 matmuls on an Ampere or newer GPU (1.0e-4 from `single_gpu` on two L40S, 1.2e-7
    on V100s), and in float32 under an in-process x64. Each worker now starts with the parent's
    `JAX_DEFAULT_MATMUL_PRECISION` and `JAX_ENABLE_X64`.
  - **`max_generations=0`** returns zero generations on `eval_only_parallel`,
    `full_pipeline_parallel` and `population_parallel_process`, as on `single_gpu`; they raised
    `UnboundLocalError`.
  - **`full_pipeline_parallel` with hidden-to-hidden connections ran out of memory** where
    `single_gpu` ran the same configuration: at depth 4 and population 100 it asked each of two or
    four GPUs for 69.48 GiB, one buffer of about P x N^2 x 401 bytes, because its pmapped pipeline
    discovered the h->h connections on every device by querying every genome's CPPN at every pair of
    grid positions. The masks and the connections are now found once per generation, outside the
    pmap, by the bounded discovery `single_gpu` runs, and replicated to the devices (sharded with
    the population on the population-parallel pipeline the tests run). The compiled temporaries at
    depth 4, population 100 and 2,048 samples went from 115.0 GiB to 2.255 GiB on two devices and
    from 114.7 GiB to 1.474 GiB on four (XLA:CPU).
- **JAX's x64 mode.** With `JAX_ENABLE_X64=1` every run stopped at its first generation, on every
  platform: TensorNEAT's species update traces two `lax.cond` calls whose branches differ in dtype
  under x64 (`check_stagnation`, float32 fitness against the species state's float64 best fitness;
  `count_members`, a float64 NaN against a float32 count), and tracing raised "cond branches must
  have equal output types". `create_config` now gives TensorNEAT's species module a `lax.cond` that
  casts both branches to their promoted dtype (`emr_hyperneat/x64_compat.py`). TensorNEAT's source
  is unchanged, and under float32 nothing is patched. Under x64 the fitness and the species
  bookkeeping are float64 (the problem's inputs stay float32). See
  [64-bit floats](docs/configuration.md#64-bit-floats).
- **`initialize` starts a new h->h cache.** In 0.1.0 only `create_config` built it, so an object
  reused for another run through `initialize` scored that run's first generations with the last
  run's connections, until the generation count passed the last refresh by `hh_refresh_interval` or
  the mask moved (the cache was on by default).
- **A bare `pytest` from the repository root** runs the package tests: a root `pytest.ini` points it
  at `emr_hyperneat/tests` and keeps it out of `third_party/`, whose TensorNEAT tests broke every
  later `import tensorneat`.
- **`depth_offload` drew another random key** before every selection step, which the standard and
  streaming paths draw only under `extra_randkey_split`, so a default depth-offload run bred another
  population from generation 1 on. It now draws the standard path's keys. `extra_randkey_split:
  true` draws 0.1.0's keys there: a feed-forward depth-offload run then reproduces 0.1.0's bit for
  bit on a CPU, and a recurrent one does with the settings that restore 0.1.0's recurrence.

### Refused (`ValueError` unless noted)

0.1.0 accepted each of these and ran without the setting, or with a default, unless noted.

- A top-level `preset`, `config_file` or `overrides` with a value, before anything else is read: they
  asked the vendored config manager for a configuration loaded from a file, and none ships, so
  0.1.0 raised `ConfigError` after two warnings. Set to None or empty they do nothing, as before.
- A genome whose discovered hidden-to-hidden set exceeds `recurrence.max_sparse_conns`, in every
  execution mode, on every route and in the substrate API (`build_recurrent_substrate`), under
  the default `sparse_cap_policy: refuse` (`SparseCapExceeded`); the message names the value that
  holds every set. 0.1.0 kept such a genome's first pairs in (source, target) position order, not
  its strongest, without a word (`sparse_cap_policy: truncate` does that, with a warning), and its
  depth offload raised `RuntimeError` at the cap.
- `hh_activation_mode: initial_only` outside `recurrence.settling_rule: hmr`,
  `dynamic_functions.global_activation`, `aggregation.global_aggregation`, and
  `cppn_output_indices.weight` other than 0.
- Dynamic functions, aggregation, neuromodulation or `settling_rule: hmr` with `depth_offload`,
  `eval_only_parallel` or, on two or more devices, `population_parallel_process`, whose evaluations
  run `tanh` nodes with sum aggregation; dynamic functions or aggregation on any
  `run_generation_multitask` call; recurrence with its two-module mode.
- LEO, a `fitness_mode` other than `mse`, `locality_radius`, `fitness_rescale_variance` and a
  `weight_threshold` other than 0.1 on depth offload, `population_parallel_process`,
  `eval_only_parallel` and the `full_pipeline_parallel` pipelines, which score `1 - MSE` on weights
  they build themselves (0.1.0 ignored the middle three there).
- An unknown `multi_gpu_strategy` (0.1.0 warned and ran the default), and the four sharding
  configurations the constructor accepted and never read (`position_config`, `island_config`,
  `hybrid_config`, `pmap_config`): the split routes read `emr_hyperneat.parallel` and the island
  model `emr_hyperneat.islands`.
- Keys nothing reads, at every level: in `emr_hyperneat` and every block under it, and at the top
  level, `algorithm_params`, `algorithm_params.emrhyperneat`, `substrate`, `swarm`, `neat` and
  `cppn` with its `genome` and `activation` blocks. The message names the right key where it is
  known. In the tests alone: `emr_hyperneat.extended` (two "recurrence" tests ran feed-forward),
  `strategy`, `pop_size`, `dynamic_functions.activation_palette`, `aggregation.aggregation_func`,
  `geometry_seeding.use_7d_inputs` (a "5-D" preset ran 7-D rows), the neuromodulation level flags
  outside mode `custom` (beside an HMR mode name, a flag that repeats its preset is accepted),
  `context_derivation` and `num_nt`. `neat.species_size` is accepted. An HMR-shaped configuration
  (`algorithm_params.hmrhyperneat`) raises instead of running every default.
  `recurrence.max_connections`, checked and never read, warns (`UserWarning`).
- Unknown activation, aggregation and interpretation names, palette entries outside
  `ACTIVATION_LIST`, an unknown `aggregation.mode` or `hh_aggregation_mode` (0.1.0 ran `tanh`,
  `sigmoid`, `sum` or `magnitude_bio` for them); a hidden activation other than `tanh` outside mode
  `global`, and an output activation other than `sigmoid` on the multi-task path, which applies a
  sigmoid; `emr_hyperneat.iteration_level` other than 1; `allow_backward`, `allow_lateral` or
  `allow_self_loops` without hidden-to-hidden connections, whether by the preset, by
  `allow_hidden_to_hidden: false` or by setting neither (0.1.0 ran feed-forward);
  `critical_periods: {enabled: true}`;
  `dynamic_functions.sparsity` outside `weight_interpretation`'s `hierarchical_sparsity`; the
  `modular` block's activation, sparsity and scaling settings outside mode `modular`, and its
  `aggregation_method` without `aggregation.mode: modular`; `palette_evolution` outside mode
  `cppn_output`, beside a `palette`, and on depth offload, the multi-device routes and
  `run_generation_multitask`; a `fitness_rescale_variance` that is not positive; `mod_neuron_ratio`
  outside (0, 1]; an aggregation palette under a mode that decodes no per-node aggregation.
- An unknown neuromodulation mode or preset (HMR's mode names translate; 0.1.0 accepted any mode
  name and raised at the first generation), a mode or preset that combines levels (`full`), a mode
  that contradicts the preset, `branch_gating: true` (never implemented), and `enabled` that
  contradicts the mode (true with no mode or preset, false with one; 0.1.0 ran the mode whatever
  `enabled` said); a `node_rule` other than `paper` outside `true_neuromodulation`; a
  `context_source` other than `'task_id'`, now its default (0.1.0 accepted `'input'` and `'learned'`
  and ignored them: context gating takes its context from the current task index).
- An unknown `emr_hyperneat.fitness_mode` (0.1.0 scored it as MSE). Multi-task `ce` and unknown
  fitness modes; settings of a `multitask` block whose `enabled` is not true; an unknown multi-task
  aggregation, generalist bonus type, two-module mode, modulation mode or activation (0.1.0 ran
  `tanh` for an unknown activation, as `evaluate_with_neurotransmitter` did);
  `orthogonality_bonus`, `specialization_bonus` and `confusion_eval_frequency` away from their
  defaults, which nothing applies. The two bonuses, `confusion_eval_frequency` and the generalist
  bonus type are checked on a `MultiTaskConfig` set by hand or by HMR's call form too.
- `enabled: false` together with `allow_hidden_to_hidden: true`, and a recurrence preset
  contradicted by `enabled` or `allow_hidden_to_hidden` in the same block (in 0.1.0 the preset took
  precedence over both keys).
- `EMRHyperNEATMultiGPU(recurrence_config=...)`. With `NotImplementedError`:
  - `run_until_threshold_with_fixed_hh`, which discovered each population slot's hidden-to-hidden
    connections once and kept them, so a genome could be evaluated with connections discovered for
    another genome;
  - the eleven runners `EMRHyperNEAT` inherited from the base class and never routed through
    (`run_until_threshold_data_parallel`, `run_until_threshold_streaming`,
    `run_until_threshold_multi_gpu` and eight more). They built and scored networks with the base
    class's pipeline instead of `EMRHyperNEAT`'s, and `run_until_threshold_data_parallel` weighted a
    padded dataset's last sample several times.

  Configure recurrence in `emr_hyperneat.recurrence`, and call `run_until_threshold`, whose
  `multi_gpu_strategy` and `execution.mode` choose the route.
- `ForwardPassMode.FULL_SPARSE`, with a `NotImplementedError` from `set_forward_mode`: it was never
  implemented, and 0.1.0 ran it with no h->h connections, as `DENSE_ONLY`.
- A settling-step count that is not an integer of at least 1 (2 under `network_model: node_list`); a
  negative `band_threshold`; `raw` with `per_genome`; levels above 1 on the `full_pipeline_parallel`
  pipelines; an explicit `streaming` or `depth_offload` mode on a multi-device route.
- `recurrence.hh_cache_enabled` on depth offload's population-chunked loop (a population above
  `execution.max_population_chunk`, or weight matrices above `execution.target_cpu_gb`): the loop
  never consulted the population-level cache, so it rediscovered the hidden-to-hidden connections
  every generation and differed from the unchunked loop. It raises at the route's entry and names
  the alternatives.

### Papers

- **The scripts whose published runs used the HMR engine are EMR versions**, since 0.2.0 does not
  ship that engine:
  - `emr-dynamic-functions-bio-inspired`: the ten `benchmark_bio_*` runners;
  - `emr-dynamic-functions`: `E3_population_sensitivity.py`, `E4_recurrent_parity8.py` and
    `E5_recurrence_type.py`;
  - `emr-hyperneat`: `run_gecco2026_experiments.py` and `run_statistical_validation.py`;
  - `emr-neuromodulation`: the shared runner `multihead_palette_neuromodulation.py` runs EMR only
    (0.1.0's `backend` parameter selected the HMR engine), with the scripts that drive it.

  Each runs its experiment as it ran: population 150 where the HMR variant ignored `neat.pop_size`,
  and iteration level 2 for the recurrent presets that ran it. It runs on EMR's defaults with HMR's
  level-0 variance rule and random-key split pinned (`scripts/runners/emr_settings.py`,
  `USE_HMR_RULE_AND_SPLIT`), and writes to `results/<dir>_emr/`, apart from the published runs.
  `run_gecco2026_experiments.py` runs `with_lateral` and `with_self` as their names say, where the
  HMR engine ran both as `hidden_only`. Each paper's README gives, run by run at one seed, what EMR
  gives against the HMR engine of release 0.1.0. The published runs reproduce on 0.1.0 with
  `scripts/replicate.py`.
- `benchmark_seed_upgrade.py` and `probe_emr_config.py` of `emr-neuromodulation`, which failed at
  import in 0.1.0, import the shared runner from its folder.
- The per-node paper's validator `scripts/analysis/_validate_emr_port.py` compares EMR with HMR's 18
  trajectories recorded on 0.1.0 (`scripts/analysis/hmr_trajectories.json`) instead of running the
  frozen module.
- Every other paper script keeps 0.1.0's code: apart from docstrings, the syntax trees of 534 of the
  555 Python files 0.1.0 ships under `papers/` are unchanged.
- `TestBioInspiredPaper`'s golden is recorded again on the EMR version of its runner
  (0.775484561920166, on arm64 and x86_64); 0.1.0's, on the HMR engine, was 0.7666119933128357.

### Documentation and text

- **Comments, docstrings, documentation and printed text**, in the package, the tests, the scripts
  and `papers/`:
  - Outside code blocks and Mermaid diagrams, Unicode arrows, `×`, `≈` and `…` are written `->`,
    `x`, `~` and `...`, and box-drawing diagrams use ASCII characters.
  - Em-dashes and ` -- ` dashes became commas, colons, parentheses or periods.
  - Rule-only comments, marker prefixes (`CRITICAL:`, `NOTE:`, `NEW:`, `FIX:`, `Key insight:`) and
    step narration are gone, except the step numbers that name the timing metric keys.
  - ALL-CAPS headings and emphasis are in sentence case, apart from acronyms, enum and mode names,
    and the published paper code's comments.
  - Notes that told how the code used to work state what it does and why; stock phrases give way to
    plain statements; British spellings are American.
  - Printed progress, verbose and log lines lose arrows, check marks, warning signs and `===`
    frames.
- **Corrected**, statements of 0.1.0 that the code, the data release or the papers' errata
  contradict:
  - **Static gating** computes one gate per hidden node from its input weights
    (`tanh(2 * mean(abs(w)))`, then `gate_scaling`), not a per-connection gate from the CPPN, which
    EMR computes only under `gate_source: cppn`.
  - **The h->h cache in the multi-GPU strategies.** The decision trees, the route banners, the
    strategy docstrings and `docs/architecture.md` give the reason to prefer `eval_only_parallel`
    for recurrent runs (it runs the full discovery, multi-hop included, on one device every
    generation), and an unscoped "Expected 1.3-1.9x speedup" is gone.
  - **Streaming's cost.** "Roughly 3-10x slower per generation" and "fastest" for the standard mode
    had no measurement behind them. The text states the modes' memory and makes no cost claim.
  - **The multi-GPU strategies:** `single_gpu` is the default and is no longer called the fastest or
    the only path the tests cover. The README states what can change results: float32 rounding in
    compaction and `truncate_variance_lookahead` (off by default).
  - **The README's speed claim** gives task, population, devices and depths and no figure. On XOR at
    population 1000, EMR-HyperNEAT on an RTX 2080 Ti is slower than a feed-forward port of
    ES-HyperNEAT on an M4 Max CPU at depths 1 to 3, and more than an order of magnitude faster per
    generation at depth 7, with the h->h cache on or off. Its runtime figure's caption says how the
    bands are computed and which points are projected, estimated or interpolated.
  - **`papers/emr-hyperneat/`**: the README no longer restates speedups without the settings they
    were measured under, and names the ES-HyperNEAT baseline as what it is, a feed-forward Python
    port of PUREPLES's algorithm. It and the docstrings of its runtime scripts say how each total is
    computed, that the per-generation series in the data release are synthetic (lognormal draws
    scaled to each run's steady time), and which cells are estimated, interpolated or projected.
    0.1.0 called every value a measured 30-generation total.
  - **The paper result notes** (`EXPERIMENT_RESULTS.md`, `BENCHMARK_RESULTS.md`) carry marked
    corrections where the data release, the runners or the papers' errata contradict them. The paper
    READMEs are corrected in place.
  - **The recurrence and the substrate in the documentation:**
    - each input settles from a zero hidden state, so no state carries between inputs;
    - the variance mask reaches a cell through its parent and keeps every reached cell;
    - `band_threshold` is the magnitude threshold for hidden-to-hidden connections.
    - The activation result is stated with its scope: in feed-forward substrates, oscillatory
      activations such as `sin` solve parity beyond XOR, which monotonic activations almost never
      do.
  - **Populations in six bio-inspired runners:** their docstrings stated populations of 200 to 800,
    passed as `neat.pop_size`. The HMR variant they ran reads `population_size`, so every published
    run used 150, as the paper's errata record.
  - **The base class `EMRHyperNEATMultiGPU`** (not exported) makes none of 0.1.0's unmeasured
    claims: depth 7 at population 1000 on 11 GiB GPUs, "the MOST EFFICIENT multi-GPU approach",
    "near-linear speedup", a faster Python loop for one generation.
  - **Palette strategies** (both paper copies): docstrings say what the code does. For example,
    strategy 11 applies its sticky rate in every phase, and strategy 140 never removes its
    guaranteed members. The catalog's total is fixed.
  - **Smaller fixes:** names of functions or files the release does not have, defaults stated wrong
    (`skip_dead_levels` is on by default, `use_smart_mutations` off), the test counts in
    `docs/testing.md`, two examples described as per-node activation that configure `global` mode,
    `fitness_utils.py`'s example, and test docstrings that credited a run through `run_generation`
    to the strategy under test.
- **The execution modes** (`docs/configuration.md`): only `depth_offload` chunks a recurrent
  evaluation over the samples, sized by `execution.recurrent_target_memory_gb`; `streaming` streams
  the CPPN queries and evaluates as `standard` does. `docs/writing-experiments.md` says what
  `run_generation`'s `skip_metrics` does in each mode (as in 0.1.0, only depth offload reads it).
- **The promise about published code** (`docs/reproducing-experiments.md`) says what holds. The code
  of a paper's runners and analysis scripts is not edited after its release, apart from the EMR
  versions above.

### Tests and CI

- New test files:
  - the substrate API, the settling rule, per-genome discovery, bucketed compaction, the pmap
    connection class and the generation metrics (`test_substrate_api.py`,
    `test_substrate_recurrent.py`, `test_recurrent_rule.py`, `test_hh_discovery_per_genome.py`,
    `test_compaction_bucketing.py`, `test_pmap_pipeline_class_constraint.py`,
    `test_generation_metrics.py`);
  - `test_replicate.py` (the replicate script; that the release map and the READMEs agree);
  - `test_text_style.py` (the patterns above stay out of the prose);
  - `test_multi_device_parity.py`: every multi-device route runs one generation on two, three and
    four forced CPU devices, and the fitness vector it passes to NEAT and the random key it returns
    must equal a single-device run's; marked slow, each split route runs five generations against
    `single_gpu`, bit for bit in ten forms, with a negative control;
  - `test_device_split.py`: the split routes' query and evaluation backends, the device grid and its
    refusals, the HMR engine's names, the one-device fallback; on two, three and four forced CPU
    devices (marked slow) every grid's query equals the one-device query bit for bit, and a
    generation with both backends equals `single_gpu`'s in eleven evaluation forms;
  - `test_islands.py`: the island seeds, every refusal, each island equal to `single_gpu` at its
    seed and population bit for bit over five generations, migration against the same islands
    without it, and on forced CPU devices (marked slow) the same islands under three placements;
  - `test_multi_gpu_worker.py` (with the workers' numeric settings), `test_paper_imports.py` (every
    script of every paper loads as `python script.py` would load it), `test_pytest_config.py`,
    `test_x64.py`, `test_pipeline_hh_outside_pmap.py`;
  - `test_silent_drops.py`: the settings above that 0.1.0 accepted and ignored raise;
  - the HMR features' tests: `test_hmr_reference.py`, `test_hmr_names.py`,
    `test_hmr_base_settings.py`, `test_unified_recipe.py`, `test_live_palette_strategy.py`,
    `test_anti_node_rule.py`, `test_custom_levels.py`, `test_leo_hmr_compound.py`,
    `test_additive_settling.py`, `test_leo_features.py`, `test_continual_learning.py`,
    `test_palette_evolution.py` and `test_node_list.py`. They read the HMR side from values recorded
    from the HMR engine on arm64 and x86_64 (`hmr_reference.py`, `hmr_reference_values.py`), never
    from a live HMR engine;
  - `test_generation_loop.py` (the traceable feed-forward generation against the general path, the
    `while_loop` against the default loop, the HMR engine's own loop through the recorded values).
- `test_isolation.py`: no module of the package and no paper or release script imports the HMR
  engine or passes `backend='hmr'`, by syntax tree, and the frozen package is not importable.
- `test_smoke.py` runs `fitness_utils.py`: its docstring's example, every fitness mode and its
  jitted form, and the threshold helper.
- `TestMaxConnectionsConfig` checks that `recurrence.max_connections` warns and changes nothing.
- Off the CPU, weights compare within ulps of `max_weight` and fitness within `1e-4`; the multi-hop
  reference check allows each weight the float32 error bound of its sums (x86 OpenBLAS sums in
  another order). Every test file sits in exactly one nightly shard (0.1.0's list missed one).
- Paper validation also covers the per-node paper's depth (E8) and Two Spirals (E7) runners. The
  `replicate` job of the paper-validation workflow, on manual dispatch, runs each release that
  `papers/releases.json` names through the script, with that release's own paper findings, and
  reruns 0.1.0's two slow classes whose runners ran the HMR engine there.
- The CPU subprocesses of the paper-validation tests and of the golden test drop an inherited
  `JAX_PLATFORM_NAME` and `JAX_BACKEND_TARGET`, which made them ask a CPU-only JAX for a GPU
  backend.
- `test_substrate_recurrent.py` compares two forwards of one substrate bit for bit on a CPU and, on
  a GPU, to float32 rounding (`rtol` 1e-4) in the genomes whose outputs one ulp in the weights does
  not move past that tolerance.
- Tests that compare runs over several generations let the genomes vary (at the default
  `genome_elitism` of 15, a population of 15 or fewer is copied unmutated); the execution-mode
  comparison checks every generation's whole fitness vector; `mode_baselines_canonical.json`
  records the modes after the key fix.

### Known limitations

- The island model runs its islands one after another from one Python thread, so their device work
  overlaps only through JAX's asynchronous dispatch. The split routes keep the LEO, multi-output and
  self-connection CPPN queries on the first device, which also holds the masks, the hidden-to-hidden
  discovery and the weight matrices of the whole population.
- On a GPU, dynamic h->h aggregation (`hh_aggregation_mode: dynamic`) does not repeat bit for bit:
  it sums each node's recurrent inputs with a scatter-add (as 0.1.0 does), which a GPU accumulates
  in no fixed order, and a genome that settles chaotically amplifies the difference (on an
  RTX 2080 Ti, up to 8.9e-4 in fitness between two evaluations of one generation). XLA's
  `--xla_gpu_deterministic_ops=true` makes it repeat, but with jax 0.6.1 it also put the substrate
  API's and the generation loop's fitness of the same genomes up to 0.5 apart, and gave a genome of
  a per-node-function configuration NaN outputs, so it is no remedy. On a CPU these runs repeat bit
  for bit.
- `population_parallel_process` evaluates each worker's shard at once, so at large populations a
  worker can run out of GPU memory (at population 1000 and depth 4, a 500-genome shard asked for
  12.72 GiB); such a worker raises.

## 0.1.0

Initial public release of EMR-HyperNEAT as a standalone, installable package.

- **Engine**: Eager Multi-Resolution HyperNEAT (eager tensor evaluation on a pre-computed
  hierarchical grid plus post-hoc variance masking), replacing ES-HyperNEAT's sequential quadtree.
  Execution modes `auto | standard | streaming | depth_offload` trade peak memory for speed and
  reach depth 11+ on a single device.
- **Evolvable computation**: per-node activation and aggregation functions, palette evolution, and
  4-vector neuromodulation for multi-behavior genotypes.
- **Substrate rule**: `variance_rule` defaults to `canonical` (one substrate across execution
  modes); `legacy` reproduces the published standard-path numbers, and the frozen `_hmr_frozen`
  module reproduces specific published tables bit-for-bit.
- **Multi-GPU**: single-device by default; `full_pipeline_parallel`, `eval_only_parallel`, and
  `streaming` distribute across devices through `run_until_threshold`.
- **Tests**: cross-layout contracts assert bitwise on CPU and to float32 rounding on a GPU;
  golden-value regressions on the pinned JAX; paper-validation tests re-run each paper's simplest
  experiment. The CPU suite runs under `pytest -n 4 --dist loadfile`.
- **Reproducibility**: the accompanying papers ship their runners; result data is a separate Zenodo
  release (DOI 10.5281/zenodo.21383893), fetched not committed.
