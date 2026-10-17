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

A key the engine does not read raises an error that names the right key when the mistake is a
known one (`emr_hyperneat.activate_time`: the settling steps are `recurrence.activate_time` or
`substrate.activate_time`) and lists the block's settings otherwise. So do an unknown name (an
activation, aggregation, interpretation, mode or preset) and a contradiction between keys. In 0.1.0
such a key or value was ignored, and the run went ahead without it. The check covers
`emr_hyperneat` and its blocks; the top level, `substrate`, `neat`, `swarm` and `cppn` are not
checked this way, and the `neat_*` blocks have their own check.

## `substrate`

| Key | Recommended | Meaning |
|---|---|---|
| `input_coords` | one `(x, y)` per input | Input node positions; **length must equal** `problem.input_shape[0]` (bias included). |
| `output_coords` | one `(x, y)` per output | Output node positions; length must equal `problem.output_shape[0]`. |
| `activate_time` | unset | Settling steps of the hidden layer when it has hidden-to-hidden connections: the same count as `recurrence.activate_time` (set one, or both to the same value; see [Feature 2](#feature-2-recurrence-memory)). No effect without hidden-to-hidden connections. |
| `hidden_activation`, `output_activation` | `tanh`, `sigmoid` | The defaults of `dynamic_functions.hidden_activation` and `output_activation` ([Feature 1](#feature-1-dynamic_functions-per-node-activation)), as in HMR's dynamic-functions variant; a name outside `ACTIVATION_FUNCTIONS` raises. 0.1.0 read neither key. |

> **Activations apply where the evaluation can apply them.** The hidden activation applies in
> `dynamic_functions` mode `global` only, so another value raises in the other modes. The output
> activation applies wherever the evaluation has a per-node function, aggregation or
> neuromodulation, so a value other than `sigmoid` raises on the plain and multi-task paths, which
> apply a sigmoid. To change the hidden activation of every node: `{"mode": "global",
> "hidden_activation": "sin"}`.

See [writing experiments](writing-experiments.md#substrate-coordinates-come-from-the-config-not-the-problem)
for the coordinate convention.

## `emr_hyperneat`: substrate discovery

These control the eager multi-resolution grid and the variance mask.

| Key | Default | Recommended | Meaning |
|---|---|---|---|
| `initial_depth` | `0` | `0` | Force full subdivision through this level: in every execution mode, every level `l <= initial_depth` holds all `4^(l+1)` positions per genome, regardless of variance. `0` forces nothing. Above `0` this is the capacity control -- `variance_threshold` alone stops subdividing at level 2-3, and selection favors smoother (lower-variance) CPPNs, so deep levels never populate on their own. |
| `variance_rule` | `canonical` | `canonical` | Level-0 subdivision variance. `canonical`: per-parent variance over the four level-1 children -- the rule every other level uses, the one streaming/depth_offload always used, and the one the papers describe; all three execution modes then compute the same substrate. `legacy`: the original standard-path rule (one scalar over level 0's own four weights, all-or-nothing subdivision of the level), which every published standard-path run used -- the six EMR paper runners pin it to reproduce those numbers bit for bit (emr-dynamic-functions E1/E2/E7/E8/E9 and the neuromodulation multihead runner; the other paper runners use the frozen HMR module, which hard-wires the legacy rule and takes no flag). Every execution mode applies either rule; in 0.1.0 streaming and depth_offload ran the canonical rule under `legacy`. |
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
| `weight_interpretation` | Per-node activation derived from the incoming weights (`interpretation`: `magnitude_bio`, the default, from their mean magnitude; `sign`; or `variance`; another name raises, which 0.1.0 ran as `magnitude_bio`), from `num_activations` or a `palette`. **Changed in 0.2.0:** with a palette, every node computes a palette function; 0.1.0 left the decoded position unmapped, so a node whose position was not itself a palette member output 0. |
| `random_fixed` / `random_generation` | Random per-node activation, fixed at init / re-rolled each generation (controls/baselines). `random_scope: genome` (the default) draws each population slot from the run's key, or from `random_seed` when set; `random_scope: shared` is 0.1.0's draw, one assignment for every genome and every run, kept to reproduce such runs. The aggregation block's random modes take the same `random_scope`. |

**The 18 activations** (`ACTIVATION_LIST`, indexable by position):

| 0 `tanh` | 1 `sigmoid` | 2 `relu` | 3 `identity` | 4 `sin` | 5 `gauss` |
|---|---|---|---|---|---|
| 6 `lelu` | 7 `softplus` | 8 `rs_adapt` | 9 `fs_fast` | 10 `lts_low` | 11 `burst` |
| 12 `resonator` | 13 `osc_adapt` | 14 `gain_mod` | 15 `receptive` | 16 `band_pass` | 17 `integrate` |

`palette` accepts an explicit list of `ACTIVATION_LIST` indices or names (e.g. `[4, 11, 12]` or
`["sin", "burst", "resonator"]`; an entry outside the list raises, which 0.1.0 accepted) or a
**named palette**:
`default`/`classification` `[0,1,2,3]`, `oscillatory` `[4,11,12,5]`, `sin_only` `[4]`,
`parity_optimal` `[4,11,12]`, `full` `[0..17]`, `bio_oscillatory` `[4,11,12,13,15]`,
`bio_adaptive` `[8,9,10,14,17]`, `phase4_all` `[13..17]`.

`index_decode` sets how the activation channel, or a weight-interpretation signal, becomes a palette
position. `tanh`, the default and 0.1.0's decode, bins `tanh` of a value that is already in `[-1, 1]`,
so the bins at both ends of a palette of 9 or more functions are never reached: with all 18, positions
2 to 15 only, and no node is ever assigned `tanh`, `sigmoid`, `band_pass` or `integrate`. `uniform`
bins the value itself, every bin equally wide, and reaches every entry. The default keeps 0.1.0's
results; another value raises. The substrate API carries it as
`SubstrateConfig.activation_index_decode`.

**On the recurrent path** (Feature 2) the hidden layer settles by one rule: from a zero state,
`activate_time` steps of `h = f(drive + h @ Whh)`, where `drive` is the input drive with the
configured aggregation (Feature 4) and `f` is everything a node does, its activation (per node,
`global`, or `tanh`) and any neuromodulation. Without hidden-to-hidden connections this is the
feed-forward evaluation. `hh_activation_mode` names the activation of the settling steps:

| `hh_activation_mode` | Settling activation |
|---|---|
| `every_iteration` (default) | The configured one, at every step. |
| `tanh` | `tanh`, whatever the mode: 0.1.0's default settling, for runs with no input aggregation. |

**Changed in 0.2.0.** The default was `initial_only`, which settled this path with `tanh`: the
input drive entered raw, every step was `tanh`, and the per-node indices were computed but not
applied. `initial_only` is now rejected with an error, as is `global_activation` (the `global`
mode reads `hidden_activation`). In 0.1.0 the recurrent path also applied no input aggregation and
used `mse` whatever the fitness mode.

`cppn_output_indices` places the extra CPPN outputs (`activation`, `aggregation`). The weight is
always channel 0, and `weight` set to anything else is rejected (0.1.0 accepted it and kept the
weight on channel 0).

Mode `modular` parses and warns that it is not implemented on the EMR path (evaluation falls back
to fixed `tanh`); its `modular` block is refused under any other mode, and the top-level `sparsity`
block, which nothing reads, is refused. `critical_periods: {"enabled": true}` raises: nothing in EMR
changes a mutation rate. `palette_evolution` is accepted only as `{"enabled": false}`, which the
per-node paper's runners carry; palette evolution is not part of EMR (the palette paper's
strategies are in `papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/`). 0.1.0
parsed all of these and only printed them.

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

`iteration_level` (default `1`, carried by every recurrent preset since 0.2.0; `2` in 0.1.0) sets how many
hops the multi-hop expansion adds (the mechanism: [architecture](architecture.md#the-recurrent-path-multi-hop-expansion-and-the-hh-cache)).
A pre-registered study of the level (1,650 runs) found level 2 better in one (problem, preset) cell of 25
and worse in 11 of its 60 tests (Holm p < 0.05), so the presets discover direct connections only; set
`iteration_level: 2` to compose them, or to reproduce a 0.1.0 run. Level 0 discovers the direct
connections as level 1 does (the multi-device pipelines run it feed-forward).
`multi_hop_second_factor` (default `masked`) sets the right operand of every multi-hop product.
`masked` composes the genome's existing direct connections, so composite connections stay within
its own positions, and the result is intersected with the preset's connection class (a forward edge
composed with a backward one would otherwise land lateral or on its own source); hop h carries
decay^(h-1). `raw`, the behavior of
0.1.0, multiplied by the raw weights over the whole union of active positions (a genome's composite
connections could end on positions it did not have active, fall outside the class, and depend on
the other genomes) and carried the decayed product into the next hop. Pass `raw` only to reproduce a
recorded run; it is refused together with `hh_discovery_mode: per_genome`.
`hh_discovery_mode` (default `per_genome`; `union` under the `raw` factor) chooses how the
candidate pairs are queried. `union` runs one
CPPN query over the population's union of active positions and then restricts each genome to its
own positions; its cost grows with population times the square of the union size. `per_genome`
queries each genome over its own active positions only, in static buckets of 8, 16, 32, ...
positions, each bucket processed in slices of a fixed number of genomes (at most 2^16 position pairs
per call, short slices padded), the whole population in one call (the union-sized population split of
the union path does not apply), so one compiled function per bucket size serves the whole run and the
cost follows the genomes' own active sets. The two modes give the same connections and weights for
every genome: on a CPU bit for bit at `iteration_level` 1 and, across 36 configurations, at level 2
(the acceptance criterion stays "same connections, weights within one float32 ulp", because the
products run over different array shapes). `per_genome` was faster than union at every depth 1 to 7
on XOR, population 1000, an RTX 2080 Ti. Dense discovery keeps
the union query whatever the setting; the pmap (multi-GPU) pipeline has its own inline discovery at
`iteration_level` 1 only and refuses a higher level on entry. A negative `band_threshold` is refused.
The discovered h→h connections can be cached between generations: `hh_cache_enabled` (default
`False` since 0.2.0, `True` in 0.1.0) switches the cache, `hh_refresh_interval` (default `5`) forces
a re-discovery every N generations, and `hh_mask_change_threshold` (default `0.1`) forces one when
the variance mask moves by more than that fraction. The cache is not a pure speed-up: the cached
set is indexed by population slot, and reproduction moves genomes between slots, so between
refreshes a genome is scored with the recurrent connections discovered for whichever genome held
its slot at the last refresh. On seed-paired XOR runs the solve rate did not change, but converged
runs lost their champion in 124 of 187 cases with the cache against 9 of 196 without it. Leave it
off unless you are reproducing a cache-on campaign. There is no per-class cache (hidden-to-hidden,
backward, lateral, self) and none is planned: the classes are masks on one CPPN query over the
active position pairs, so a cached class would be stale for a changed genome exactly as the whole
set is. The per-class controls are `allow_hidden_to_hidden`, `allow_backward`, `allow_lateral` and
`allow_self_loops`, which decide what is discovered. `max_sparse_conns` (default 10000) sizes the
padded connection arrays and truncates a genome with more valid connections than the cap, in flat
position order (its first pairs, not its strongest); it is a memory setting, not a speed setting.
A truncation warns once per process, and each generation's metrics count the genomes it cut
(`hh_truncated_genomes`, `hh_max_uncapped`); 0.1.0 truncated without a word.

**Settling steps.** A recurrent hidden layer settles for `activate_time` steps (the rule under Feature 1).
Set the count in the `recurrence` block or as `substrate.activate_time`; if both are set they must be
equal, or `create_config` raises `ValueError` naming both. Unset, it is the preset's own count
(`full_recurrent`: 20) or `2 (2**max_depth + 1)` (18 at depth 3). With hidden-to-hidden connections on,
an explicit count must be an integer of at least 1: a bool, a float or a string is refused, here and in
`SubstrateConfig`. Without them nothing settles, so any value is accepted and has no effect.
`set_recurrence_config(config)` sets the forward mode, `iteration_level` and the steps by the same rule:
the config's `activate_time`, else an explicit `substrate.activate_time`, else the derived count (called
before `create_config`, when the depth is not known yet, it derives nothing). In 0.1.0 a recurrent
configuration replaced `substrate.activate_time` with its own count, and `set_recurrence_config`
kept the previous step count and `iteration_level`.

### Reproducing a 0.1.0 recurrent run

0.2.0 changed the recurrence defaults. These keys restore 0.1.0's, under a preset or explicit
settings:

```python
"emr_hyperneat": {
    "recurrence": {
        "preset": "hidden_only",           # the run's own preset or settings
        "iteration_level": 2,              # 0.1.0's level
        "multi_hop_second_factor": "raw",  # 0.1.0's multi-hop algebra
        "hh_discovery_mode": "union",      # 0.1.0's query
        "hh_cache_enabled": True,          # 0.1.0's h->h connection cache
    },
    "execution": {"compaction_bucket": "exact"},  # 0.1.0's array shapes, for x86 and GPUs
}
```

Six more depend on the configuration:

- **`enabled: false` with `allow_hidden_to_hidden: true`**: set `enabled: true`. 0.2.0 refuses the
  pair; 0.1.0 ran it recurrent in the standard mode and feed-forward under `depth_offload`.
- **A preset block that also sets `enabled` or `allow_hidden_to_hidden` against the preset**: drop the
  key. 0.2.0 refuses the contradiction; in 0.1.0 the preset took precedence over both keys.
- **Per-node or `global` activations**: set `dynamic_functions.hh_activation_mode: tanh`. 0.1.0's
  default, `initial_only`, settled every step with `tanh` instead.
- **`substrate.activate_time`** in a recurrent configuration: drop it. 0.1.0 settled the preset's
  own count or `2 (2**max_depth + 1)` in its place.
- **`fitness_mode`** other than `mse` on the recurrent or neuromodulated path: set `mse`, which 0.1.0
  used there whatever the key said.
- **`true_neuromodulation`** in a single-task run: set `neuromodulation.node_rule: hmr_full`, the rule
  0.1.0's single-task path computed. Its multi-task path computed the default, `paper`.

On a CPU, in the standard execution mode, these settings reproduce 0.1.0's fitness bit for bit. No
setting restores the rest: the recurrent path applies input aggregation, which 0.1.0 did not apply
there; 0.1.0's neuromodulated recurrent evaluators used a residual update and its
`eval_only_parallel` evaluator an older rule; the `full_pipeline_parallel` pipelines apply the
preset's connection class (0.1.0 evaluated every preset there as full recurrence without self
loops) and refuse a level above 1; and a NaN fitness on the standard path becomes `-inf` before
selection. A run whose CPPN had aggregation as
its only `cppn_output` feature also needs the channel pin under [Feature 4](#feature-4-aggregation).

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
| `modulatory_neurons` | A dedicated modulatory neuron type (Soltoggio-style): `mod_neuron_ratio` (0.1) of each genome's own active positions, at least one, those of highest input-weight variance, whose mean activity scales the genome's nodes (EMR has no plasticity for it to gate, unlike Soltoggio's). In 0.1.0 every genome took the same positions, that fraction of every grid position. |
| `true_neuromodulation` | Neurotransmitter vectors `[DA, 5HT, NE, ACh]` + receptor densities. This is the mode the multi-task path uses; `num_nt_types` is usually 4. |

Under `true_neuromodulation` the first three channels of the vector modulate a node through its
receptors, `m = receptors[:3] @ nt[:3]` with strength `s`, and `node_rule` (default `paper`) chooses
the node: `paper`, the published rule, applies the gain `clip(base_gain + s m, 0.1, 5)` and the bias
`s m` inside the activation and the gate `sigmoid(m)` after it; `hmr_full`, HMR FULL's rule, which
0.1.0's single-task path computed, applies the gain `base_gain + s m` with no bias and the gate
`sigmoid(5 m)`. `modulation_mode: gating_only` applies the gate alone (`sigmoid(s m)` under `paper`)
and `gain_bias_only` drops the gate. The fourth channel inverts the outputs when
`use_output_inversion` is set and the vector has four or more channels. The multi-task two-module mode computes the paper's rule only and
refuses `hmr_full`.

A `preset` (`static_gating`, `xdg_style`, `modulatory_only`, `true_neuromodulation`,
`true_neuromodulation_4nt`, `true_neuromodulation_5nt`, or `disabled`) sets the mode of its level
(`xdg_style` is `context_gating`) and its parameters, which the block's own keys override. In 0.1.0
a preset left the mode `disabled`, so a run configured by a preset alone had no modulation. A preset
that combines levels (`full`), an unknown preset or mode, a `mode` that contradicts the preset,
`branch_gating: true` (never implemented), and `enabled` that contradicts the mode (true with no mode
or preset, false with one) raise. The block's other keys raise too: the level flags
(`static_gating: true` and the others; one mode is one level), `gate_from_cppn` (EMR derives its gates
from the incoming weights) and `context_derivation` (never read; the context comes from
`context_source`) were ignored in 0.1.0.

Neuromodulation composes with dynamic functions because they act on different things: dynamic
functions make a per-node *structural* choice (which activation), neuromodulation applies a
per-node *parametric* modulation (how much gain and gating). Both can be enabled at once.

For multi-task, the companion `multitask` sub-dict drives `run_generation_multitask`. Its keys:
`enabled` (the other keys raise unless it is true), `num_tasks` (default `2`), `task_names`,
`nt_per_task` (one NT vector per task; omitted = named-task presets), `fitness_aggregation` ∈
`mean`/`min`/`weighted`/`product`/`softmin`/`harmonic` (default `mean`), `task_weights` (for
`weighted`), `softmin_temperature` (default `0.1`), `joint_evolution` (default `True`),
`fitness_mode` (one of `MULTITASK_FITNESS_MODES`, default `mse`; `ce` raises), `hidden_activation`
and `per_task_activation` (names of `ACTIVATION_FUNCTIONS`; 0.1.0 ran `tanh` for an unknown one),
`modulation_mode` (`full`, `gating_only`, `gain_bias_only`) and `modulation_strength_override`,
`two_module_mode` (`none`, `parallel`, `sequential`), and the generalist bonus
(`generalist_bonus_type` ∈ `none`/`min_bonus`/`variance_penalty`/`threshold_bonus`,
`generalist_bonus_weight`, `generalist_threshold`). In 0.1.0 the parser dropped `softmin_temperature`,
`modulation_mode`, `modulation_strength_override`, `two_module_mode` and the generalist bonus, which
acted only when set on a `MultiTaskConfig` object, and two-module mode always fell back to one module.
Eight keys are accepted only at their default, because nothing in EMR applies them:
`orthogonality_bonus`, `orthogonality_bonus_weight` and `specialization_bonus` (default `0.0`; each
adds one constant to every genome, which never changes a rank), `specialization_bonus_weight` and
`modulation_penalty` (default `0.0`), `confusion_eval_frequency` (default `0`), `branch_gating_mode`
(default `none`) and `orthogonality_metric` (default `cosine_mean`). `run_generation_multitask`
evaluates each task with one hidden activation (`per_task_activation` or `hidden_activation`) and
sum aggregation, so it refuses the `dynamic_functions` and `aggregation` modes, and its two-module
mode, which is feed-forward, refuses recurrence. A worked example is in
[writing experiments](writing-experiments.md#multi-task-one-substrate-several-tasks); the runners
under `papers/emr-neuromodulation/scripts/runners/` are the full reference.

## Feature 4: `aggregation`

Aggregation is how a hidden node combines its inputs. The functions are
`sum, mean, max, min, product, maxabs`; another `global_function`, an unknown `mode`, an
`interpretation` other than `magnitude_bio`, `variance`, `sign_uniformity` or `fan_in`, and an
`hh_aggregation_mode` other than `sum` or `dynamic` raise (0.1.0 ran `sum` or `magnitude_bio`). One
function for every node:

```python
"aggregation": {"mode": "global", "global_function": "max"}
```

Per node, from the CPPN, is `"mode": "cppn_output"` (below); `weight_interpretation` derives it from
the incoming weights. The bio-inspired palette paper's runners
(`papers/emr-dynamic-functions-bio-inspired/scripts/runners/`) evolve activation palettes; the
aggregation palettes their dual strategies compute are recorded and reach no evaluation (the one
runner that sets them as `agg_palette` changes nothing, since no engine reads it; see the
CHANGELOG's known limitations). **Changed in 0.2.0:** the key
`global_aggregation` is rejected (the `global` mode reads `global_function`). In 0.1.0 it was
accepted with no effect, so a `global` configuration that used it ran `sum`, which is why the 0.1.0
version of this page described global aggregation as having no standalone effect. Aggregation now
applies on the recurrent path too, and with `weight_interpretation` its hidden-to-hidden side
follows the configured interpretation (0.1.0 ran `magnitude_bio` there for `sign_uniformity` and
`fan_in`).

With `"aggregation": {"mode": "cppn_output"}` the CPPN gets a dedicated output for the
aggregation choice, read at each hidden node's self-connection. **Changed in 0.2.0:** when
aggregation is the only `cppn_output` feature, the CPPN has two outputs and aggregation now reads
the second. Before 0.2.0 it read channel 0, the weight, and the second output was evolved and never
read. To reproduce runs made that way, pin the old channel:
`"dynamic_functions": {"cppn_output_indices": {"aggregation": 0}}`.

`aggregation.index_decode` (`tanh`, the default, or `uniform`) does for the aggregation channel and
the weight interpretations what `dynamic_functions.index_decode` does for activations, the
hidden-to-hidden side of `hh_aggregation_mode: dynamic` included (in the substrate API,
`SubstrateConfig.aggregation_index_decode`). Six functions are all reachable under both decodes,
but under `tanh` their bins are unequal: over the channel's range, `sum` and `maxabs` get about 10%
each, `max` and `min` about 17%, `mean` and `product` about 23%; under `uniform`, a sixth each.

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
| `seed_weight` | `-1.0` | Stored and printed, and read by nothing: it changes no result (in 0.1.0 and 0.2.0 alike). |

## Other `emr_hyperneat` keys

| Key | Default | Meaning |
|---|---|---|
| `fitness_mode` | `'mse'` | Fitness computation: `mse`, `accuracy`, `soft_accuracy`, `bce`, `hybrid`, `acc_mse` (accuracy plus 0.01 of 1 - MSE), `accuracy_confident` (accuracy plus 0.1 of the outputs' mean distance from 0.5) or `ce` (cross-entropy over softmax outputs), on every single-task evaluation path (in 0.1.0 the recurrent and neuromodulated paths always used `mse`). An unknown name raises; 0.1.0 scored it, and `acc_mse` and `accuracy_confident`, as MSE. Classification problems use the accuracy-style modes; `mse` is what the golden tests lock. A NaN fitness, from an overflowing network or a NaN target, becomes `-inf` before selection, and the generation's metrics report `nan_fitness_count`. |
| `multi_gpu_strategy` | unset (`single_gpu`) | Case-insensitive string selecting the multi-GPU strategy: `"single_gpu"`, `"full_pipeline_parallel"`, `"streaming"`, `"eval_only_parallel"`, or `"population_parallel_process"`. Nine older alias names are accepted and mapped onto those five, here and in the constructor's `strategy` argument alike; an unknown value raises (0.1.0 warned and kept the default). `population_parallel_process` is not recommended. `eval_only_parallel` and `population_parallel_process` evaluate `tanh` nodes with sum aggregation and no neuromodulation, so they refuse dynamic functions, aggregation and neuromodulation. What each one parallelizes, and the fact that it engages only through `run_until_threshold`: [architecture](architecture.md#what-the-multi-gpu-strategies-parallelize). |
| `extra_randkey_split` | `False` | `True` reproduces the frozen HMR module's per-generation PRNG split, for bit-exact replication of the results recorded on it. Details: `papers/emr-dynamic-functions/REPRODUCIBILITY.md`. |
| `locality_radius` | unset | Scales each input-to-hidden and hidden-to-output weight by `min(1, radius / distance)` between its endpoints, so distant connections are weakened. |
| `fitness_rescale_variance` | unset | Scores `max(0, 1 - MSE / variance)` instead of `1 - MSE`, which spreads the fitness of targets with a small variance; must be positive (0.1.0 ignored a value that is not). |

Depth offload, `population_parallel_process`, `eval_only_parallel` and the `full_pipeline_parallel`
pipelines score `1 - MSE` on weights they build themselves, so they refuse a `fitness_mode` other
than `mse`, `locality_radius` and `fitness_rescale_variance` (0.1.0 ignored them there).

## Execution mode

Controls **where a generation's intermediate arrays live**, not what is computed. All three modes
produce identical results for the configurations each accepts, under either `variance_rule`; they
trade speed for peak memory. (In 0.1.0 streaming and depth_offload ignored `variance_rule: legacy`
and ran the canonical rule.) This is what decides whether a given depth runs at all.

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
| `depth_offload` | Assembled in host RAM (or a memmap), moved to the device one chunk at a time | Depth 11+, where the matrices exceed device memory outright. Depth 11 is 22.4M candidate positions, about 380 MB per genome. Evaluates `tanh` nodes with sum aggregation and no neuromodulation, so it refuses dynamic functions, aggregation and neuromodulation (0.1.0 evaluated without them), and it records no champion for `evaluate_on_data`. |
| `auto` | — | Picks by `max_depth` against the two thresholds. On a GPU backend the pick is also memory-aware: a `standard` selection escalates to `depth_offload` when the standard path's calibrated working-set estimate exceeds the device — population size can push `standard` past device memory at depths the thresholds alone would allow. Inert on CPU; never overrides an explicit mode. A configuration `depth_offload` cannot evaluate is refused when `auto` selects it, at `create_config` or, for the memory-aware escalation, at the first generation; set `streaming` explicitly for it. |

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
| `compaction_bucket` | `"auto"` | Shape of the compacted recurrent evaluation space. `auto` pads the union of reached positions to a bucketed length with inert virtual positions, so the length repeats across generations and each length compiles once per run (a count that lands exactly on a bucket length takes the same path); results equal `exact` bit for bit on an arm64 CPU, and to float32 rounding on x86 CPUs and GPUs (a padded sum can round differently: a fitness by an ulp or two on x86). `exact` evaluates on exactly the union, as 0.1.0 did. Random per-node draw modes and depth offload always use exact shapes. |
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

The NEAT operators read three blocks beside `neat` in `algorithm_params.emrhyperneat`, each over
the defaults shown; an unknown setting raises:

```python
"neat_mutation":  {"conn_add_prob": 0.5, "conn_delete_prob": 0.5,
                   "node_add_prob": 0.2, "node_delete_prob": 0.2},
"neat_species":   {"compatibility_threshold": 3.0, "max_stagnation": 20, "species_elitism": 15},
"neat_selection": {"genome_elitism": 15, "survival_threshold": 0.2},
```

**Changed in 0.2.0.** 0.1.0 applied these defaults whatever a configuration set; drop the blocks to
reproduce a 0.1.0 run that set them. Other NEAT settings, such as `species_size` in the `neat` block,
are not read.

## Keys that parse but do not act

The keys below are accepted and change nothing, or accepted only at the value that changes nothing;
they are listed so silence is never mistaken for effect. Every other key the engine does not read
raises.
When in doubt, copy a config from a paper runner under `papers/*/scripts/` — those are known-good.

| Key | Where it sits | Status |
|---|---|---|
| `species_size` | `neat` | Not read; the NEAT operators read `neat_mutation`, `neat_species` and `neat_selection` ([above](#neat--population)). |
| `palette_evolution: {enabled: false}` | `dynamic_functions` | Accepted for the per-node paper's runners; any other value raises. |
| `critical_periods` (`phase1_end`, `phase2_end`, `min_plasticity`), `enabled: false` | `dynamic_functions` | Parsed and stored, read by nothing; `enabled: true` raises. The critical periods of the palette paper are part of its strategies (`papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/`). |
| `modular` and its `sparsity` | `dynamic_functions` | Mode `modular` warns that it is not implemented; under another mode the block raises. |
| `division_threshold` | `emr_hyperneat` | Warns (`DeprecationWarning`) and names the live controls. |
| `iteration_level: 1` | `emr_hyperneat` | The multi-hop level is `recurrence.iteration_level`; another value raises. |
| `seed_weight` | `geometry_seeding` | Read by nothing ([above](#geometry_seeding-shipped-not-part-of-any-published-paper)). |
| `use_vectorized_discovery` | `recurrence` | Retired: it was stored and never consulted. |
| `orthogonality_bonus`, `orthogonality_bonus_weight`, `specialization_bonus`, `specialization_bonus_weight`, `modulation_penalty`, `confusion_eval_frequency`, `branch_gating_mode`, `orthogonality_metric` | `multitask` | Accepted at their defaults only ([Feature 3](#feature-3-neuromodulation-multi-task)). |

Next: [reproducing the paper experiments](reproducing-experiments.md), or back to
[writing experiments](writing-experiments.md).
