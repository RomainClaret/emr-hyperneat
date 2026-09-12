# Configuration reference

Everything is one nested dict passed to `EMRHyperNEAT.create_config(...)`. The fixed outer keys are
`algorithm_params` → `emrhyperneat`; under that sit four sections:

```python
config = {"algorithm_params": {"emrhyperneat": {
    "population_size": 150,        # how many genomes per generation
    "substrate":     { ... },      # where inputs/outputs sit + default activations
    "emr_hyperneat": { ... },      # substrate-discovery knobs + the four feature sub-dicts
    "neat":          { ... },      # population size (other NEAT knobs are fixed; see below)
}}}
```

You can omit almost everything and rely on defaults. The tables below list both the parser's
**Default** (what you get when the key is absent) and the **Recommended** value (what the papers
use) — where they differ, set the key explicitly.

## `substrate`

| Key | Recommended | Meaning |
|---|---|---|
| `input_coords` | one `(x, y)` per input | Input node positions; **length must equal** `problem.input_shape[0]` (bias included). |
| `output_coords` | one `(x, y)` per output | Output node positions; length must equal `problem.output_shape[0]`. |

> **Activations are not set on the substrate.** The `substrate.output_activation` / `hidden_activation`
> keys are inert (ignored). The defaults are `tanh` (hidden) / `sigmoid` (output); set
> them through the `dynamic_functions` feature below instead: `{"mode": "global", "hidden_activation":
> "sin"}` for the hidden activation, and `"output_activation": "..."` for the output activation.

See [writing experiments](writing-experiments.md#substrate-coordinates-come-from-the-config-not-the-problem)
for the coordinate convention.

## `emr_hyperneat`: substrate discovery

These control the eager multi-resolution grid and the variance mask.

| Key | Default | Recommended | Meaning |
|---|---|---|---|
| `initial_depth` | `0` | `0` | Force full subdivision through this level: in every execution mode, every level `l <= initial_depth` holds all `4^(l+1)` positions per genome, regardless of variance. `0` forces nothing. Above `0` this is the capacity control -- `variance_threshold` alone stops subdividing at level 2-3, and selection favors smoother (lower-variance) CPPNs, so deep levels never populate on their own. |
| `variance_rule` | `canonical` | `canonical` | Level-0 subdivision variance. `canonical`: per-parent variance over the four level-1 children -- the rule every other level uses, the one streaming/depth_offload always used, and the one the papers describe; all three execution modes then compute the same substrate. `legacy`: the pre-0.4 standard-path rule (one scalar over level 0's own four weights, all-or-nothing subdivision of the level), which every published standard-path run used -- the six EMR paper runners pin it to reproduce those numbers bit for bit (emr-dynamic-functions E1/E2/E7/E8/E9 and the neuromodulation multihead runner; the other paper runners use the frozen HMR module, which hard-wires the legacy rule and takes no flag). |
| `max_depth` | `1` | `2`–`3` | Maximum grid depth. Candidate positions grow ~4× per level (depth 2 → 84, 3 → 340, 4 → 1364). Higher depth = larger substrate = slower. |
| `variance_threshold` | `0.03` | `0.03` | Minimum CPPN-output variance for a region to survive the mask. Lower = denser substrate. |
| `division_threshold` | -- | -- | Read by nothing on the eager multi-resolution path (canonical ES-HyperNEAT subdivides on it; the eager path subdivides on `variance_threshold` and `initial_depth`). Supplying it raises a `DeprecationWarning` naming the live controls. |
| `band_threshold` | `0.3` | `0.3` | Band-pruning threshold for connection expression (internal, leave at default). |
| `max_weight` | `8.0` | `8.0` | Scales CPPN output to a connection weight (internal, leave at default). |
| `verbose` | `False` | `False` | Print substrate-build logging. |

`band_threshold` and `max_weight` are substrate-discovery internals. The examples omit them and
use these defaults; change `max_depth`, `variance_threshold`, and (for deep substrates)
`initial_depth` first.

The four feature sub-dicts (`dynamic_functions`, `aggregation`, `neuromodulation`, `recurrence`) also
live under `emr_hyperneat` and are documented next.

## Feature 1: `dynamic_functions` (per-node activation)

Lets each node use a different activation, instead of one fixed `hidden_activation`. This is the most
impactful feature for parity/periodic problems.

```python
"dynamic_functions": {"mode": "cppn_output", "palette": "oscillatory"}
```

| `mode` | What it does |
|---|---|
| `disabled` | Fixed `tanh` hidden / `sigmoid` output (the baseline). |
| `global` | All hidden nodes use one activation: set `"hidden_activation": "sin"` (or any name below). |
| `cppn_output` | The CPPN emits a per-node activation index from a palette: it becomes a multi-output network, one channel for the weight and one for the index. Set `num_activations` (e.g. 4 or 6) or a `palette` (below). |
| `weight_interpretation` | Per-node activation derived from the incoming weight's `sign` / `magnitude` / `variance` (set `interpretation`). |
| `random_fixed` / `random_generation` | Random per-node activation, fixed at init / re-rolled each generation (controls/baselines). |

**The 18 activations** (`ACTIVATION_LIST`, indexable by position):

| 0 `tanh` | 1 `sigmoid` | 2 `relu` | 3 `identity` | 4 `sin` | 5 `gauss` |
|---|---|---|---|---|---|
| 6 `lelu` | 7 `softplus` | 8 `rs_adapt` | 9 `fs_fast` | 10 `lts_low` | 11 `burst` |
| 12 `resonator` | 13 `osc_adapt` | 14 `gain_mod` | 15 `receptive` | 16 `band_pass` | 17 `integrate` |

`palette` accepts an explicit index list (e.g. `[4, 11, 12]`) or a **named palette**:
`default`/`classification` `[0,1,2,3]`, `oscillatory` `[4,11,12,5]`, `sin_only` `[4]`,
`parity_optimal` `[4,11,12]`, `full` `[0..17]`, `bio_oscillatory` `[4,11,12,13,15]`,
`bio_adaptive` `[8,9,10,14,17]`, `phase4_all` `[13..17]`.

## Feature 2: `recurrence` (memory)

Adds hidden-to-hidden connections so the network can carry state. Use a preset:

```python
"recurrence": {"preset": "hidden_only"}
```

| Preset | h→h | backward | lateral | self-loops |
|---|---|---|---|---|
| `feedforward` (default) | – | – | – | – |
| `hidden_only` | ✓ | – | – | – |
| `with_backward` | ✓ | ✓ | – | – |
| `with_lateral` | ✓ | – | ✓ | – |
| `with_self` | ✓ | – | – | ✓ |
| `full_recurrent` | ✓ | ✓ | ✓ | ✓ |

`iteration_level` (default `2`) sets how many propagation hops the multi-hop expansion adds (the
mechanism: [architecture](architecture.md#the-recurrent-path-multi-hop-expansion-and-the-hh-cache)).
The discovered h→h connections are cached between generations: `hh_cache_enabled` (default `True`)
switches the cache, `hh_refresh_interval` (default `5`) forces a re-discovery every N generations,
and `hh_mask_change_threshold` (default `0.1`) forces one when the variance mask moves by more than
that fraction.

## Feature 3: `neuromodulation` (multi-task)

Neurotransmitter vectors modulate node behavior, which lets one substrate serve several tasks. This
feature is required by `run_generation_multitask`.

```python
"neuromodulation": {"enabled": True, "mode": "true_neuromodulation", "num_nt_types": 4}
```

| `mode` | What it does |
|---|---|
| `disabled` | No modulation. |
| `static_gating` | The CPPN emits a per-connection gate in `[0, 1]`. |
| `context_gating` | A task context vector modulates the gates (XdG-style). |
| `modulatory_neurons` | A dedicated modulatory neuron type (Soltoggio-style). |
| `true_neuromodulation` | Neurotransmitter vectors `[DA, 5HT, NE, ACh]` + receptor densities. This is the mode the multi-task path uses; `num_nt_types` is usually 4. |

Neuromodulation composes with dynamic functions because they act on different things: dynamic
functions make a per-node *structural* choice (which activation), neuromodulation applies a
per-node *parametric* modulation (how much gain and gating). Both can be enabled at once.

For multi-task, the companion `multitask` sub-dict drives `run_generation_multitask`. Its keys:
`enabled`, `num_tasks` (default `2`), `task_names`, `nt_per_task` (one NT vector per task; omitted
= named-task presets), `fitness_aggregation` ∈ `mean`/`min`/`weighted`/`product`/`softmin`/`harmonic`
(default `mean`), `task_weights` (for `weighted`), `joint_evolution` (default `True`),
`orthogonality_bonus` / `specialization_bonus` (default `0.0`), `hidden_activation`, and
`per_task_activation`. A worked example is in
[writing experiments](writing-experiments.md#multi-task-one-substrate-several-tasks); the runners
under `papers/emr-neuromodulation/scripts/runners/` are the full reference.

## Feature 4: `aggregation` (partial)

Per-node aggregation (how a node combines its inputs) is present in the code but **only exercised as
part of the bio-inspired palette meta-learning**, not as a standalone knob. In particular, the
standalone "global aggregation" presets do not change behavior through a single config key, so they
are not documented as user controls here. If you need evolved aggregation, follow the bio-inspired
palette runners under `papers/emr-dynamic-functions-bio-inspired/scripts/runners/`, which co-evolve
activation *and* aggregation palettes. The aggregation functions themselves are
`sum, mean, max, min, product, maxabs`.

## `geometry_seeding` (shipped, not part of any published paper)

A locality bias on connection weights, after the geometry-seeding idea in Risi & Stanley (2012):
enabling it switches the CPPN from its 5-input form to a 7-input form that also receives the
source→target coordinate offsets, so weight patterns can condition on relative geometry. It is
off by default, covered by `tests/test_geometry_seeding.py`, and **no published result depends on
it** (see the scope note in the README).

```python
"emr_hyperneat": {
    "geometry_seeding": {"enabled": True, "seed_weight": -1.0},
}
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Use the 7-input CPPN with the locality bias. Changes substrate discovery and results. |
| `seed_weight` | `-1.0` | Strength of the seeded locality bias. |

## Other `emr_hyperneat` keys

| Key | Default | Meaning |
|---|---|---|
| `fitness_mode` | `'mse'` | Fitness computation on the dynamic-functions evaluation path: `mse`, `accuracy`, `soft_accuracy`, `bce`, or `hybrid`. Classification problems use the accuracy-style modes; `mse` is what the golden tests lock. |
| `multi_gpu_strategy` | unset (`single_gpu`) | Case-insensitive string selecting the multi-GPU strategy: `"single_gpu"`, `"full_pipeline_parallel"`, `"streaming"`, `"eval_only_parallel"`, or `"population_parallel_process"`. Nine older alias names are accepted and mapped onto those five; unknown values warn and keep the default. `population_parallel_process` is not recommended. What each one parallelizes, and the fact that it engages only through `run_until_threshold`: [architecture](architecture.md#what-the-multi-gpu-strategies-parallelize). |
| `extra_randkey_split` | `False` | `True` reproduces the frozen HMR module's per-generation PRNG split, for bit-exact replication of the results recorded on it. Details: `papers/emr-dynamic-functions/REPRODUCIBILITY.md`. |

## Execution mode

Controls **where a generation's intermediate arrays live**, not what is computed. Under the default
`variance_rule: canonical` all three modes produce identical results; they trade speed for peak
memory. (Under `variance_rule: legacy` the standard path uses the pre-0.4 level-0 rule and disagrees
with the other two, which is why setting `legacy` outside `standard` now warns.) This is what decides whether a given
depth runs at all.

```python
"emr_hyperneat": {
    "execution": {
        "mode": "auto",       # auto | standard | streaming | depth_offload
        "depth_threshold": 9,           # auto picks depth_offload at/above this
        "streaming_depth_threshold": 8, # auto picks streaming at/above this
    }
}
```

| Mode | Where the arrays live | When you need it |
|---|---|---|
| `standard` (default) | All on device | Depth 7 and below. Fastest, highest peak memory. This is the path the published results and the golden tests were produced with. |
| `streaming` | One hierarchical level at a time | Depth 8+. Roughly 3-10x slower per generation, and the difference between running and an OOM. |
| `depth_offload` | Assembled in host RAM (or a memmap), moved to the device one chunk at a time | Depth 11+, where the matrices exceed device memory outright. Depth 11 is 22.4M candidate positions, about 380 MB per genome. |
| `auto` | — | Picks by `max_depth` against the two thresholds. On a GPU backend the pick is also memory-aware: a `standard` selection escalates to `depth_offload` when the standard path's calibrated working-set estimate exceeds the device — population size can push `standard` past device memory at depths the thresholds alone would allow. Inert on CPU; never overrides an explicit mode. |

**Omitting the `execution` block gives `standard`.** `enable_streaming: true` is a valid spelling
of `mode: streaming`.

### Memory budgets

Defaults are usually right; reach for these when a run OOMs or when you want it to use less.

| Key | Default | Meaning |
|---|---|---|
| `target_cpu_gb` | `32.0` | Host RAM budget for the weight matrices. Sets the population chunk size. |
| `max_population_chunk` | `None` | Override the computed genomes-per-chunk. |
| `target_shard_gb` | `0.5` | Device memory per position shard during CPPN queries. Sized on the CPPN forward's while-loop carry (2 x `max_nodes` floats per genome-position), the measured dominant term. |
| `max_lookahead_positions` | `5000000` | Memory budget for the variance lookahead. Levels larger than it are computed in parent-row slabs -- bounded memory, identical substrate. |
| `skip_dead_levels` | `True` | Skip weight processing for levels with zero reached positions. Changes cost, never results (asserted per pair). Default `True`; the reported runtimes were measured with it off, so the baseline capture pins `False`; the GECCO timing runners use the frozen HMR path, which never reads the key. Acts on the `depth_offload` path only. |
| `target_eval_shard_gb` | `1.0` | Device memory per shard during evaluation. |
| `cppn_pos_chunk_size` | `100000` | Upper clamp on positions per CPPN query. |
| `eval_chunk_levels` | `3` | Levels loaded at once during evaluation. |
| `eval_chunk_positions` | `1000000` | Maximum positions per device chunk. |
| `recurrent_target_memory_gb` | `2.0` | Peak budget for recurrent evaluation, which chunks population *and* samples. |
| `use_cpu_grid` | `False` | Keep the grid structure itself in host RAM. Needed from depth 12, where the grid alone exceeds device memory. |
| `use_memmap` / `memmap_path` | `True` / `/tmp/emr_memmap` | Back large chunks with memory-mapped files. |
| `enable_disk_offload` / `disk_offload_path` | `False` / `/tmp/emr_offload` | Spill to disk at extreme depth. |
| `verbose` | `False` | Per-level statistics. |

### One knob that changes results

`truncate_variance_lookahead` (default `false`). By default `max_lookahead_positions` is a pure
memory budget: oversized levels have their variance computed in exact parent-row slabs, so any
budget yields the identical substrate. Setting `truncate_variance_lookahead: true` zeroes the
variance of oversized levels instead, which halts subdivision at and below them, so **the
discovered substrate changes**. Every other key above leaves results untouched.

One related knob lives outside the `execution` block: `emr_hyperneat.population_chunk_size`.
`-1` (the default) is AUTO -- a budget gate that returns the full population unchanged whenever
the largest level's query carry fits `target_shard_gb`, and chunks only where the unchunked query
would OOM. Per-genome query results are independent, so chunking never changes values.

## `neat` / population

In the EMR path the `neat` block reads only the population size:

```python
"neat": {"pop_size": 150}     # or set top-level "population_size"
```

The remaining NEAT hyperparameters (mutation rates, species/compatibility thresholds, elitism) are
fixed inside the engine and are not read from this block, so, for example, `species_size` placed
here has no effect. If you need to change those, you are into engine territory, not configuration.

## Keys that parse but do not act

A few keys are accepted by the parser (or by a sub-dict) without being wired end-to-end. Setting
them raises no error and changes nothing; they are listed so silence is never mistaken for effect.
When in doubt, copy a config from a paper runner under `papers/*/scripts/` — those are known-good.

| Key | Where it sits | Status |
|---|---|---|
| aggregation "global" presets | `aggregation` | No standalone effect; evolved aggregation runs through the bio-palette runners (see Feature 4). |
| `species_size` | `neat` | Not read; NEAT hyperparameters are fixed inside the engine. |
| `palette_evolution.enabled` | `emr_hyperneat` | Handled by the bio-inspired meta-learning strategies, not by this parser. |
| `critical_periods` (`enabled`, `phase1_end`, `phase2_end`, `min_plasticity`) | `dynamic_functions` | Parsed and stored, read by nothing in this class. The critical-period behavior in the PPSN paper is implemented inside its strategy library (`papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/`). |
| `two_module_mode`, `generalist_bonus_type` (and the related bonus weights) | `multitask` | Silently dropped by the config parser. They act only when a `MultiTaskConfig` is constructed programmatically with them set; `run_generation_multitask` then honors them. |

Next: [reproducing the paper experiments](reproducing-experiments.md), or back to
[writing experiments](writing-experiments.md).
