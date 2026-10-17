# Changelog

Notable changes to the released `emr_hyperneat` package. The version tracks
`emr_hyperneat/_version.py`.

## 0.2.0

Results of recurrent runs change: one settling rule, a multi-hop expansion within each genome's own
connections, and new defaults.
[docs/configuration.md](docs/configuration.md#reproducing-a-010-recurrent-run) lists the settings
that reproduce a 0.1.0 recurrent run. On a CPU, the per-node paper's runners give the same results
as with 0.1.0; the neuromodulation paper's runner departs from 0.1.0 on its default EMR backend, because
0.2.0 applies the `neat_species` block it passes
([docs/reproducing-experiments.md](docs/reproducing-experiments.md#choosing-a-release)).

### Added

- **Substrate API** (`emr_hyperneat.substrate`), the functions the generation loop builds its
  substrate with: `population_substrates`, `substrate_queries`, `build_substrate` and
  `substrate_forward` for the feed-forward substrate; `hh_query_rows`, `build_recurrent_substrate` and
  `query_recurrent_substrate`, which build the loop's hidden-to-hidden connections bit for bit; and
  `return_states=True` for the settling states. `substrate_config()` refuses what it cannot describe.
- **Per-genome h->h discovery** (`recurrence.hh_discovery_mode`): each genome is queried over its own
  active positions in static buckets, and gets the union query's connections and fitness, bit for bit
  on a CPU.
- **Bucketed compaction** (`execution.compaction_bucket`): the recurrent evaluation pads its active
  set to a bucketed length, so a length compiles once per run. Results equal `exact` bit for bit on
  an arm64 CPU; on x86 a padded sum can round differently, moving a fitness by a float32 ulp or two.
- **Records**: `run_until_threshold` returns its `route`, `devices` and `execution_mode`; per-genome
  counts of the last h->h discovery in the module-level `emrhyperneat._last_discovery_stats`,
  including each genome's count before the `max_sparse_conns` cap and what the cap dropped;
  `nan_fitness_count`; `hh_truncated_genomes` and `hh_max_uncapped` in the generation's metrics. A
  truncation by the cap warns once per process (0.1.0 truncated without a word). The generation's
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
  (`node_rule: hmr_full`). In 0.1.0 the single-task path computed HMR FULL's rule (an unclipped gain,
  no bias, the gate `sigmoid(5 m)`) and the multi-task path the published one (a gain clipped to
  [0.1, 5], an additive bias, the gate `sigmoid(m)`). The output inversion acts only when
  `use_output_inversion` is set and the vector has four channels or more: 0.1.0's multi-task path read
  the fourth channel whatever the count, so with two or three channels a modulation channel also
  inverted the outputs, and it ignored the flag.
- **`hh_activation_mode: every_iteration`**. 0.1.0's default, `initial_only`, is refused; `tanh`
  settles as it did when no input aggregation is configured.

### Fixed

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
- **`fitness_mode` applies on the recurrent and neuromodulated paths** (in 0.1.0 these paths used
  `mse`), and a NaN fitness becomes `-inf` before selection on the standard path too.
- **Multi-task**: `multitask.fitness_mode` is read and `accuracy_confident` is computed (in 0.1.0
  both evaluated `mse`), and the generation counter advances once per generation (twice in 0.1.0).
- **`neat_mutation`, `neat_species` and `neat_selection` reach TensorNEAT**, over the defaults 0.1.0
  always used; an unknown setting raises. In 0.1.0 a configuration's values for them were not applied
  (drop the blocks to reproduce such a run).
- **`weight_interpretation` with a `palette` gives every node a palette function**, as `cppn_output`
  and the random modes do. In 0.1.0 the decoded palette position was not mapped to the function, so a
  node whose position was not itself a palette member output 0.
- **A strategy name passed to the constructor takes the route it has as `multi_gpu_strategy`**
  (`STRATEGY_NAME_MAP`); `EMRHyperNEAT()` still runs `single_gpu`. In 0.1.0 only `streaming`,
  `eval_only_parallel` and `population_parallel_process` kept their multi-device route through the
  constructor, and every other name ran the one-device loop under the requested name.
- **A neuromodulation preset runs its level**: it sets the mode (`xdg_style` is `context_gating`), and
  `use_self_connection_query` reaches it. In 0.1.0 a preset left the mode `disabled`, so a run
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
  dropped them, so they acted only when set on a `MultiTaskConfig` object, and two-module mode read a
  cached grid that nothing set and always fell back to one module.
- **`fitness_mode: acc_mse` and `accuracy_confident` on the single-task paths** compute what the
  multi-task evaluators compute (accuracy plus 0.01 of 1 - MSE; accuracy plus 0.1 of the outputs' mean
  distance from 0.5). In 0.1.0 both names computed MSE there.
- **`substrate.hidden_activation` and `output_activation` are the defaults of the `dynamic_functions`
  activations**, as in HMR's dynamic-functions variant; 0.1.0 read neither key.
- **With `weight_interpretation` aggregation the hidden-to-hidden side follows the interpretation**;
  0.1.0 ran `magnitude_bio` there for `sign_uniformity` and `fan_in`.
- **`variance_rule: legacy` applies on every execution mode.** In 0.1.0 streaming and depth_offload
  computed the canonical level-0 variance under `legacy`. A configured mode warned, but a mode chosen
  at run time (`auto`'s memory-aware escalation on a GPU) did not, so such a run used the canonical
  rule silently. The configuration warning is gone.
- **Depth offload reports the mean, min and std fitness over the finite values** and counts NaN
  fitness in `nan_fitness_count`, as the standard path does; in 0.1.0 a NaN fitness, told as `-inf`,
  made its mean fitness `-inf`.
- **`evaluate_on_data` scores the evolved champion** (in 0.1.0 it scored a dense `tanh` network
  rebuilt from the champion).
- **An aggregation-only `cppn_output` CPPN reads its second output**; 0.1.0 read the weight output
  there, which `cppn_output_indices: {aggregation: 0}` reproduces.
- **The multi-GPU worker's cache path runs**; in 0.1.0 it built `EMRConfig` with a field that class
  does not have, which raised an error.
- **The recurrent evaluation no longer recompiles its settling loop every generation** when its node
  function is one the engine names (the default `tanh`, a global activation, aggregation-only runs).
  Each recompiled loop kept its compiled code for the rest of the run: on a Linux CPU (JAX 0.6.1), a
  recurrent XOR run at depth 3 and population 100 grew by about 2 MiB per generation, to 3,612 MiB
  after 1,000 generations; it now holds near 1,400 MiB after the first 150. Results are unchanged.
- **`depth_offload` follows the standard mode's recurrence rule**: it settles hidden-to-hidden
  connections whenever the configuration allows them. In 0.1.0 it also required `enabled` and
  evaluated iteration level 0 feed-forward; level 0 now discovers direct connections, as in the
  standard mode.

### Refused (`ValueError`)

- `hh_activation_mode: initial_only`, `dynamic_functions.global_activation`,
  `aggregation.global_aggregation`, and `cppn_output_indices.weight` other than 0.
- Dynamic functions, aggregation or neuromodulation with `depth_offload`, `eval_only_parallel` or,
  on two or more devices, `population_parallel_process`, whose workers evaluate `tanh` nodes with sum
  aggregation; dynamic functions or aggregation with `run_generation_multitask`; recurrence with its
  two-module mode.
- An unknown `multi_gpu_strategy` (0.1.0 warned and ran the default); a constructor `strategy` with
  no route in `EMRHyperNEAT` (`hybrid`, `island_model` and the other base-class strategies), and the
  four sharding configurations the constructor accepted and never read (`position_config`,
  `island_config`, `hybrid_config`, `pmap_config`).
- Keys EMR does not read, in `emr_hyperneat` and every block under it, naming the right key where it
  is known (0.1.0 ignored them): in the tests alone `emr_hyperneat.extended` (two "recurrence" tests
  ran feed-forward), `strategy`, `pop_size`, `dynamic_functions.activation_palette`,
  `aggregation.aggregation_func`, `geometry_seeding.use_7d_inputs` (a "5-D" preset ran 7-D rows), the
  neuromodulation level flags (`static_gating: true` and the others; one mode is one level),
  `gate_from_cppn`, `context_derivation` and `num_nt`. An HMR-shaped configuration
  (`algorithm_params.hmrhyperneat`) raises instead of warning and running every default.
- Unknown activation, aggregation and interpretation names, palette entries outside
  `ACTIVATION_LIST`, an unknown `aggregation.mode` or `hh_aggregation_mode` (0.1.0 ran `tanh`, `sigmoid`,
  `sum` or `magnitude_bio` for them); a hidden activation other than `tanh` outside mode `global`, and
  an output activation other than `sigmoid` on the plain and multi-task paths (those evaluations do
  not apply them);
  `emr_hyperneat.iteration_level` other than 1; `allow_backward`, `allow_lateral` or
  `allow_self_loops` without hidden-to-hidden connections (0.1.0 ran feed-forward);
  `critical_periods: {enabled: true}`, the top-level `dynamic_functions.sparsity` block and the
  `modular` block outside mode `modular`, which nothing reads; `palette_evolution` other than
  `{enabled: false}`; a `fitness_rescale_variance` that is not positive.
- A `fitness_mode` other than `mse`, `locality_radius` and `fitness_rescale_variance` on depth
  offload, `population_parallel_process`, `eval_only_parallel` and the `full_pipeline_parallel`
  pipelines, which score `1 - MSE` on weights they build themselves (0.1.0 ignored the three there).
- An unknown neuromodulation mode or preset, a preset that combines levels (`full`), a mode that
  contradicts the preset, `branch_gating: true` (never implemented), and `enabled` that contradicts the
  mode (true with no mode or preset, false with one; 0.1.0 ran the mode whatever `enabled` said).
- An unknown `emr_hyperneat.fitness_mode` (0.1.0 scored it as MSE). Multi-task `ce` and unknown fitness
  modes; settings of a `multitask` block whose `enabled` is not true; an unknown multi-task
  aggregation, generalist bonus type, two-module mode, modulation mode or activation (0.1.0 ran `tanh`
  for an unknown activation, as `evaluate_with_neurotransmitter` did); the multi-task settings nothing
  applies (the two constant bonuses, `orthogonality_bonus_weight`, `specialization_bonus_weight`,
  `modulation_penalty`, `confusion_eval_frequency`, `branch_gating_mode` and `orthogonality_metric`)
  away from their defaults.
- `enabled: false` together with `allow_hidden_to_hidden: true`, and a recurrence preset contradicted
  by `enabled` or `allow_hidden_to_hidden` in the same block (in 0.1.0 the preset took precedence
  over both keys).
- `EMRHyperNEATMultiGPU(recurrence_config=...)`, and `run_until_threshold_with_fixed_hh` with a
  `NotImplementedError`: that runner discovered each population slot's hidden-to-hidden connections
  once and kept them, so a genome could be evaluated with connections discovered for another genome.
  Configure recurrence in `emr_hyperneat.recurrence`.
- `ForwardPassMode.FULL_SPARSE`, with a `NotImplementedError` from `set_forward_mode` and
  `forward_unified_extended`: it was never implemented, and 0.1.0 ran it with no h->h connections, as
  `DENSE_ONLY`.
- A settling-step count that is not an integer of at least 1; a negative `band_threshold`; `raw` with
  `per_genome`; levels above 1 on the `full_pipeline_parallel` pipelines; an explicit `streaming` or
  `depth_offload` mode on a multi-device route.

### Tests and CI

- New tests for the substrate API, the settling rule, per-genome discovery, bucketed compaction and
  the pmap connection class; the global-aggregation tests use the key the engine reads (0.1.0's used
  `global_aggregation`, which the engine does not read, so they ran `sum`).
- Off the CPU, weights compare within ulps of `max_weight` and fitness within `1e-4`; the multi-hop
  reference check allows each weight the float32 error bound of its sums (x86 OpenBLAS sums in
  another order). Every test file sits in exactly one nightly shard (0.1.0's list missed one).
- Paper validation also covers the per-node paper's depth (E8) and Two Spirals (E7) runners.
  `test_replicate.py` checks the replicate script, and that the release map and the READMEs agree;
  the `replicate` job of the paper-validation workflow, on manual dispatch, runs each release that
  `papers/releases.json` names through the script, with that release's own paper findings.
- The CPU subprocesses of the paper-validation tests and of the golden test drop an inherited
  `JAX_PLATFORM_NAME` and `JAX_BACKEND_TARGET`. When the suite ran on a GPU from a shell profile that
  sets them to `gpu`, they asked JAX for a GPU backend that `JAX_PLATFORMS=cpu` never loads, and
  failed before the paper code ran.
- `test_substrate_recurrent.py` compares two forwards of one substrate bit for bit on a CPU and, on
  a GPU, to float32 rounding (`rtol` 1e-4) in the genomes whose outputs one ulp in the weights does
  not move past that tolerance. With dynamic h→h aggregation a GPU's forward differs from itself
  (below), and in a genome that settles chaotically by much more than rounding, so the bit-for-bit
  and all-genome comparisons failed there.

### Known limitations

- On a GPU, dynamic h→h aggregation (`hh_aggregation_mode: dynamic`) does not repeat bit for bit: it
  sums each node's recurrent inputs with a scatter-add (as 0.1.0 does), which a GPU accumulates in
  no fixed order, and a genome that settles chaotically amplifies the difference (on an RTX 2080 Ti,
  up to 8.9e-4 in fitness between two evaluations of one generation). XLA's
  `--xla_gpu_deterministic_ops=true` makes it repeat, but with jax 0.6.1 it also put the substrate
  API's and the generation loop's fitness of the same genomes up to 0.5 apart, and gave a genome of
  a per-node-function configuration NaN outputs, so it is no remedy. On a CPU these runs repeat bit
  for bit.
- Per-node activations, neuromodulation, multi-task evaluation, `eval_chunk_size` chunking and the
  self-connection query of `cppn_output` configurations still recompile every generation; the fix
  above shows what that costs for the settling loop.
- A runner that sets `agg_palette` between generations, as it sets `df_palette`, changes nothing: no
  engine in the package reads it. The one runner of the release that sets it,
  `papers/emr-dynamic-functions-bio-inspired/scripts/runners/benchmark_bio_nonparity_cl.py`, drives
  the frozen HMR engine, which does not read it either. An `aggregation.palette` setting raises as an
  unknown key.

## 0.1.0

Initial public release of EMR-HyperNEAT as a standalone, installable package.

- **Engine**: Eager Multi-Resolution HyperNEAT — eager tensor evaluation on a pre-computed
  hierarchical grid plus post-hoc variance masking, replacing ES-HyperNEAT's sequential quadtree.
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
