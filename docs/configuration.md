# Configuration reference

Everything is one nested dict passed to `EMRHyperNEAT.create_config(...)`. The fixed outer keys are
`algorithm_params` -> `emrhyperneat`; under that sit four sections:

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
use). Where they differ, set the key explicitly.

A key the engine does not read in `emr_hyperneat` or one of its blocks raises an error that names
the right key when the mistake is a known one (`emr_hyperneat.activate_time`: the settling steps are
`recurrence.activate_time` or `substrate.activate_time`) and lists the block's settings otherwise.
So do an unknown name (an activation, aggregation, interpretation, mode or preset) and a
contradiction between keys. In 0.1.0 such a key or value was ignored, and the run went ahead without
it. The `neat_*` blocks raise on an unknown setting too, and so do the levels outside these: the top
level, `algorithm_params`, `algorithm_params.emrhyperneat`, `substrate`, `neat`, `swarm` and `cppn`
(with its `genome` and `activation` blocks), where 0.1.0 also ignored such a key without a word.
`neat.species_size` is accepted
([below](#keys-that-parse-but-do-not-act)). A configuration without its `emr_hyperneat` block warns
that every setting takes its default and is not checked further.

## `substrate`

| Key | Recommended | Meaning |
|---|---|---|
| `input_coords` | one `(x, y)` per input | Input node positions; **length must equal** `problem.input_shape[0]` (bias included). |
| `output_coords` | one `(x, y)` per output | Output node positions; length must equal `problem.output_shape[0]`. |
| `activate_time` | unset | Settling steps of the hidden layer when it has hidden-to-hidden connections: the same count as `recurrence.activate_time` (set one, or both to the same value; see [Feature 2](#feature-2-recurrence-hidden-to-hidden-connections)). No effect without hidden-to-hidden connections. |
| `hidden_activation`, `output_activation` | `tanh`, `sigmoid` | The defaults of `dynamic_functions.hidden_activation` and `output_activation` ([Feature 1](#feature-1-dynamic_functions-per-node-activation)), as in HMR's dynamic-functions variant; a name outside `ACTIVATION_FUNCTIONS` raises. 0.1.0 read neither key. |

> **Activations apply where the evaluation can apply them.** The hidden activation applies in
> `dynamic_functions` mode `global` only, so another value raises in the other modes. The output
> activation applies everywhere except the multi-task path, which applies a sigmoid, so a value other
> than `sigmoid` raises there. On the plain path (no per-node function, aggregation or
> neuromodulation) a configured output activation runs as mode `global` with a `tanh` hidden
> activation, which equals the plain evaluation bit for bit, as HMR's dynamic-functions variant
> applied it (0.1.0 ignored it on this path and applied a sigmoid). To change the hidden
> activation of every node: `{"mode": "global", "hidden_activation": "sin"}`.

See [writing experiments](writing-experiments.md#substrate-coordinates-come-from-the-config-not-the-problem)
for the coordinate convention.

## `emr_hyperneat`: substrate discovery

These control the eager multi-resolution grid and the variance mask.

| Key | Default | Recommended | Meaning |
|---|---|---|---|
| `initial_depth` | `0` | `0` | Force full subdivision through this level: in every execution mode, every level `l <= initial_depth` holds all `4^(l+1)` positions per genome, regardless of variance. `0` forces nothing. Above `0` this is the capacity control: `variance_threshold` alone stops subdividing at level 2-3, and selection favors smoother (lower-variance) CPPNs, so deep levels never populate on their own. |
| `variance_rule` | `canonical` | `canonical` | Level-0 subdivision variance. `canonical`: per-parent variance over the four level-1 children (the rule every other level uses, the one streaming/depth_offload always used, and the one the papers describe); all three execution modes then compute the same substrate. `legacy`: the original standard-path rule (one scalar over level 0's own four weights, all-or-nothing subdivision of the level), which every published standard-path run used. The six EMR paper runners pin it to reproduce those numbers bit for bit (emr-dynamic-functions E1/E2/E7/E8/E9 and the neuromodulation multihead runner). The runners whose published runs used the frozen HMR module, which hard-wires the legacy rule and takes no flag, run EMR from 0.2.0 and pin it as well, with `extra_randkey_split`, through their paper folder's `scripts/runners/emr_settings.py`. Every execution mode applies either rule; in 0.1.0 streaming and depth_offload ran the canonical rule under `legacy`. |
| `max_depth` | `1` | `2` to `3` | Maximum grid depth. Candidate positions grow ~4x per level (depth 2 -> 84, 3 -> 340, 4 -> 1364). Higher depth = larger substrate = slower. |
| `variance_threshold` | `0.03` | `0.03` | Minimum CPPN-output variance for a region to survive the mask. Lower = denser substrate. |
| `division_threshold` | - | - | Read by nothing on the eager multi-resolution path (canonical ES-HyperNEAT subdivides on it; the eager path subdivides on `variance_threshold` and `initial_depth`). Supplying it raises a `DeprecationWarning` naming the live controls. |
| `band_threshold` | `0.3` | `0.3` | Magnitude threshold for hidden-to-hidden connections: a pair is a connection when its weight's magnitude exceeds it (no band test runs, despite the name; input-to-hidden and hidden-to-output weights use `weight_threshold`). |
| `weight_threshold` | `0.1` | `0.1` | Pruning threshold of the input-to-hidden and hidden-to-output weights: a weight whose magnitude is at or below it is pruned (`WEIGHT_THRESHOLD`, the default). A number in `[0, max_weight)`. The standard and streaming builds and the substrate API read it; LEO and the swarm build prune at 0.1, so another value raises with them, as it does on depth offload and the multi-device pipelines, which build their own weights. 0.1.0 pruned at 0.1 always and ignored the key. |
| `max_weight` | `8.0` | `8.0` | Scales CPPN output to a connection weight (internal, leave at default). |
| `verbose` | `False` | `False` | Print substrate-build logging. |

`band_threshold`, `weight_threshold` and `max_weight` are substrate-discovery internals. The
examples omit them and use these defaults; change `max_depth`, `variance_threshold`, and (for deep
substrates) `initial_depth` first.

`sparse_forward_threshold` (an integer of at least -1), `fuse_w1_computation` and `skip_unused_masks`
(bools) are HMR's base-engine memory and speed settings. The base class parses them, and nothing
reads them: this class computes the reached-position mask alone and narrows the evaluation by
compaction, so no value changes a result. Their values are checked (0.1.0 read them as given).

The four feature sub-dicts (`dynamic_functions`, `aggregation`, `neuromodulation`, `recurrence`) also
live under `emr_hyperneat` and are documented next.

## Feature 1: `dynamic_functions` (per-node activation)

Lets each node use a different activation, instead of one fixed `hidden_activation`. Oscillatory
activations such as `sin` solve feed-forward parity beyond XOR, which monotonic activations almost
never do: on Parity-4, `sin` reaches the 0.95 threshold in 30 of 30 runs and nine of the ten
monotonic functions in none (recurrence changes this).

```python
"dynamic_functions": {"mode": "cppn_output", "palette": "oscillatory"}
```

| `mode` | What it does |
|---|---|
| `disabled` | Fixed `tanh` hidden / `sigmoid` output (the baseline). |
| `global` | All hidden nodes use one activation: set `"hidden_activation": "sin"` (or any name below). |
| `cppn_output` | The CPPN emits a per-node activation index from a palette: it becomes a multi-output network, one channel for the weight and one for the index. Set `num_activations` (e.g. 4 or 6) or a `palette` (below). |
| `weight_interpretation` | Per-node activation derived from the incoming weights (`interpretation`: `magnitude_bio`, the default, from their mean magnitude; `sign`; `variance`; and HMR's `magnitude` (`magnitude_bio` inverted), `combined` (magnitude, variance and the sign ratio together), `hierarchical_sparsity`, `sparsity_threshold` and `stp_inspired`, which select the activation as `magnitude_bio` does and add a sparsity or a weight scaling at evaluation (below); another name raises; 0.1.0 ran every other name, HMR's five included, as `magnitude_bio`), from `num_activations` or a `palette`. **Changed in 0.2.0:** with a palette, every node computes a palette function; 0.1.0 left the decoded position unmapped, so a node whose position was not itself a palette member output 0. |
| `modular` | HMR's three orthogonal layers, set in the `modular` block: `activation_method` (`magnitude_bio`, the default, `magnitude`, `sign`, `variance`, or `none` for `tanh` everywhere), `sparsity_method` (`none`, the default, `hierarchical`, `threshold` or `wta`; feed-forward evaluation only) and `scaling_method` (`none`, the default, or `stp_inspired`). 0.1.0 parsed the mode, warned that it was not implemented and evaluated fixed `tanh`. |
| `random_fixed` / `random_generation` | Random per-node activation, fixed at init / re-rolled each generation (controls/baselines). `random_fixed` keeps a function per population slot and grid position, also when hidden-to-hidden connections narrow the evaluation to each generation's active positions (0.1.0 failed there once the active count changed). `random_scope: genome` (the default) draws each population slot from the run's key, or from `random_seed` when set; `random_scope: shared` is 0.1.0's draw, one assignment for every genome and every run, kept to reproduce such runs. The aggregation block's random modes take the same `random_scope`, and its `random_fixed` keeps a function per slot and position in the same way. |

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
| `initial_only` | Under `recurrence.settling_rule: hmr` only (Feature 2): the configured activation at the first step, `tanh` at the later ones, HMR's mode. |

**Changed in 0.2.0.** The default was `initial_only`, which settled this path with `tanh`: the
input drive entered raw, every step was `tanh`, and the per-node indices were computed but not
applied. `initial_only` is now rejected with an error outside HMR's settling rule, as is
`global_activation` (the `global` mode reads `hidden_activation`). In 0.1.0 the recurrent path also applied no input aggregation and
used `mse` whatever the fitness mode.

`cppn_output_indices` places the extra CPPN outputs (`activation`, `aggregation`). The weight is
always channel 0, and `weight` set to anything else is rejected (0.1.0 accepted it and kept the
weight on channel 0). The CPPN's output count is `num_cppn_outputs` on the algorithm object
(`cppn_num_outputs`, HMR's name, reads the same).

The `sparsity` block sets the fraction of each sample's active values a grid level keeps under
`hierarchical_sparsity` (`level_0` 0.05, `level_1` 0.20, `level_2_plus` 0.40), and `modular.sparsity`
the same for the `hierarchical` layer, plus `wta_k_percent` (0.10) for `wta`. Each fraction must be
in (0, 1], and a block the configuration would not read raises (0.1.0 read neither block). The
sparsity layers count a genome's active nodes only, where HMR counted every position of the
evaluation axis. The `modular` block's activation, sparsity and scaling settings act under mode
`modular` only and raise under another; its `aggregation_method` is read by `aggregation.mode:
modular` (Feature 4).

`cppn_read` and `weight_source` are HMR's reads, opt-ins for runs that reproduce HMR's.
`cppn_read: input_mean` averages a node's function channel over its input-to-node queries, where
EMR reads the node's self-connection (`self`, the default); it needs mode `cppn_output`, a
dedicated channel and the standard execution mode. `weight_source: raw` has the weight
interpretations read the raw weight channel, where EMR reads `W1`, scaled and thresholded (`w1`,
the default). The `aggregation` block takes the same two keys.

`palette_evolution` is HMR's random walk on the palette, built in. After `stagnation_generations`
(5) generations without a new best fitness, every function's bit flips with probability
`mutation_rate` (0.1), and a result with fewer than `min_active` (2) functions keeps the old mask.
The walk starts from `initial_mask` (`full`, the default, `default`, or a list of indices or names;
a `palette` beside it raises) and needs mode `cppn_output`, whose decode it owns: `decode: stable`
(the default) bins the channel over the whole list and snaps a disabled function to its nearest
enabled one, so a toggle changes only the nodes of its bin; `rebin`, HMR's, bins over the enabled
functions, so every toggle remaps a quarter to a half of every genome's nodes. It steps in the
per-generation loop, so the paths that do not run that loop (`run_generation_multitask`, depth
offload, the multi-device pipelines) refuse it. 0.1.0 ignored the block.

`live_strategy` is HMR's live palette strategy: a runner's strategy object sets the palette after
each generation. Pass it with `set_palette_strategy(strategy)` before the first generation; the object
has `get_active_palette()` (`ACTIVATION_LIST` indices) and may have `post_generation_update(...)` and
`initialize()`. When a later palette is longer, a share of the population (`perturbation_rate`,
default 0.15) gets `perturbation_strength` (0.3) added to the bias of the CPPN's activation output
node, which moves the activation channel toward the higher bins; `perturbation_decay` (0) lowers the
rate as the run goes, to `rate * max(0, 1 - decay * generation / 100)`. HMR added the strength to
TensorNEAT's node-key column, not
to a bias, and split the run's key to choose the genomes; here the genomes are drawn from a key folded
from `state.randkey`, which stays as it is. It needs mode `cppn_output` and refuses
`palette_evolution`, which also sets the palette. Calling `set_palette_strategy` without the block
enables HMR's defaults. 0.1.0 ignored the key and had no `set_palette_strategy`.

`critical_periods: {"enabled": true}` raises: nothing in EMR changes a mutation rate (the palette
paper's critical periods are part of its strategies,
`papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/`). 0.1.0 parsed it and
ignored it.

## Feature 2: `recurrence` (hidden-to-hidden connections)

Adds hidden-to-hidden connections: each input settles the hidden layer over the settling steps,
starting from a zero hidden state, so no state carries between inputs. Use a preset:

```python
"recurrence": {"preset": "hidden_only"}
```

| Preset | h->h | backward | lateral | self-loops |
|---|---|---|---|---|
| `feedforward` (default) | no | no | no | no |
| `hidden_only` | yes | no | no | no |
| `with_backward` | yes | yes | no | no |
| `with_lateral` | yes | no | yes | no |
| `with_self` | yes | no | no | yes |
| `full_recurrent` | yes | yes | yes | yes |

`iteration_level` (default `1`, carried by every recurrent preset since 0.2.0; `2` in 0.1.0) sets how many
hops the multi-hop expansion adds (the mechanism: [architecture](architecture.md#the-recurrent-path-multi-hop-expansion-and-the-h-h-cache)).
A pre-registered study of the level (1,650 runs) found level 2 better in one (problem, preset) cell of 25
and worse in 11 of its 60 tests (Holm p < 0.05), so the presets discover direct connections only; set
`iteration_level: 2` to compose them, or to reproduce a 0.1.0 run. Level 0 discovers the direct
connections as level 1 does, on every path (0.1.0's multi-device pipelines and depth offload ran
it feed-forward).
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
The discovered h->h connections can be cached between generations: `hh_cache_enabled` (default
`False` since 0.2.0, `True` in 0.1.0) switches the cache, `hh_refresh_interval` (default `5`) forces
a re-discovery every N generations, and `hh_mask_change_threshold` (default `0.1`) forces one when
the variance mask moves by more than that fraction. `hh_cache_trigger` (default `time_or_mask`) says
what refreshes it: `time` refreshes every `hh_refresh_interval` generations only, as HMR's unified
model did; it is refused when the cache is off, and `hh_mask_change_threshold` is refused beside
`time`. `initialize` starts a new cache, so an object reused for another run does not score it with
the last run's connections (0.1.0 kept them until the next refresh). The cache is not a pure
speed-up: the cached
set is indexed by population slot, and reproduction moves genomes between slots, so between
refreshes a genome is scored with the recurrent connections discovered for whichever genome held
its slot at the last refresh. On seed-paired XOR runs the solve rate did not change, but converged
runs lost their champion in 124 of 187 cases with the cache against 9 of 196 without it. Leave it
off unless you are reproducing a cache-on campaign. There is no per-class cache (hidden-to-hidden,
backward, lateral, self) and none is planned: the classes are masks on one CPPN query over the
active position pairs, so a cached class would be stale for a changed genome exactly as the whole
set is. The per-class controls are `allow_hidden_to_hidden`, `allow_backward`, `allow_lateral` and
`allow_self_loops`, which decide what is discovered.

**The hidden-to-hidden cap.** `max_sparse_conns` (default 10000) sizes the padded connection
arrays; it is a memory setting, not a speed setting. `sparse_cap_policy` decides what happens to a
genome whose discovered set exceeds it, in every execution mode, on every route and in the
substrate API:

- `refuse`, the default, raises `SparseCapExceeded`, a `ValueError` whose message names the value
  that holds every set. Its attributes say which genomes: `uncapped` (each genome's count before
  the cap, in population or batch order), `over` (the indices over the cap) and `largest`, so a
  caller can leave those genomes out.
- `truncate` keeps the genome's first `max_sparse_conns` pairs in flat position order, not its
  strongest, as 0.1.0 did without a word (except under depth offload, which raised at the cap). A
  `SparseCapWarning` reports the first truncation of a set that large.
- `grow` raises the cap for the rest of the run to the largest set plus a quarter, rounded up to a
  multiple of 1,024, and discovers the generation again at that width, so no genome is truncated.
  A `SparseCapWarning` names the value to set next time; each growth compiles the new width once.
  `max_sparse_conns_ceiling` bounds it (no ceiling by default; a set beyond it raises
  `SparseCapExceeded`). `grow` cannot be combined with `hh_cache_enabled`, whose cached sets keep
  the width they were found in.

`set_sparse_cap(max_sparse_conns=..., sparse_cap_policy=..., max_sparse_conns_ceiling=...)` changes
them from the next generation on. The substrate API reads `SubstrateConfig.sparse_cap_policy` and
`max_sparse_conns_ceiling`; under `grow`, the returned substrate's configuration holds the cap it
was built at. Each generation's metrics report the largest set (`hh_max_uncapped`), the genomes
truncated (`hh_truncated_genomes`, 0 unless `truncate`) and the cap the discovery used
(`hh_max_sparse_conns`). The result of `run_until_threshold` carries `sparse_cap`: the cap the run
ended with, which its champion was evaluated under, the configured cap, the policy, the ceiling,
the largest set, the discoveries that truncated and each growth (`[from, to]`). Save it with the
champion, so the network rebuilds the same.

Deep recurrent runs can need far more than the default. Of 200 runs at depth 5, population 150 and
`iteration_level` 2 with union discovery (the five recurrence presets on four problems, 10 seeds
each), 164 had a genome over 10,000 at some generation, and the largest set held 358,008
connections. Set a cap of that order for such runs, for example `max_sparse_conns: 360000`; if a
genome still exceeds it, the refusal names the value its generation needed. `sparse_cap_policy:
grow` finds the cap as the run goes, at the cost of a compilation per growth. The default stays
10,000 because the slots cost memory in every run, 13 bytes per genome per slot (about 700 MB at
population 150 for 360,000), and because HMR's settling rule, the additive rule and dynamic
hidden-to-hidden aggregation also scatter over every slot at each settling step.

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

**Settling rule.** `settling_rule` (default `emr`) chooses the rule. `hmr` is the rule of HMR's
dynamic-functions variants: `h = f(x W1)`, then `activate_time - 1` steps of `h = g(h + recurrent(h))`,
where `g` is `tanh` under `hh_activation_mode: initial_only` and the configured function under
`every_iteration`. The input drive enters at the first step only, and the `h +` term acts as a unit
self-connection on every hidden node, so the two rules give different networks from generation 0. The
cross-problem control (GECCO Table 3), the per-node paper's recurrent tables and the PPSN topology
check ran HMR's rule; `hmr` reproduces it. It needs hidden-to-hidden recurrence, and it is refused
on the paths that build their own weights (depth offload and the multi-device pipelines). With
neuromodulation it runs HMR FULL's additive rule, the one HMR's neuromodulation benchmark ran: the
node of the input drive, then `activate_time` steps of `h = h + update(recurrent(h))`, with the
recurrent input summed. The update is `tanh(gates * r)` under static and context gating and the node
rule with `tanh` under true neuromodulation; on the multi-task path it is the node itself, with the
task's activation. It refuses modulatory neurons (FULL's modulatory recurrence read one mask for the
whole population, and no runner used it), per-node hidden-to-hidden aggregation, and an
`every_iteration` activation other than `tanh` (set `hh_activation_mode: initial_only`).

**HMR's unified model (UNI) as a recipe.** UNI discovered direct hidden-to-hidden connections,
settled them by HMR's rule and refreshed a cache on a timer. EMR runs it with:

```python
"emr_hyperneat": {
    "weight_threshold": band_threshold,   # UNI pruned W1 and W2 at band_threshold
    "variance_rule": "legacy",
    "extra_randkey_split": True,
    "recurrence": {
        "iteration_level": 1, "use_dense_discovery": True, "settling_rule": "hmr",
        "hh_cache_enabled": True, "hh_cache_trigger": "time",  # UNI's default cache
    },
}
```

For UNI's default block (`allow_hidden_to_hidden` alone, `enabled` false), set `enabled: true` with
`allow_lateral`: UNI's connection classes there are forward and lateral pairs without self-loops,
and EMR refuses `enabled` false with hidden-to-hidden connections on. Three UNI defects are not
reproduced: padded active indices entered discovery as position 0, discovery ran over the
population's union of active positions instead of each genome's own, and a per-chunk cap truncated
the pairs; the recipe equals UNI where none of them acts.

### HMR's node-list model

HMR's recurrent-network variant discovered each genome's network on the leaves of a complete
quadtree and settled it as a list of nodes. EMR runs it as a second network model, opt-in:

```python
"emr_hyperneat": {
    "max_depth": 2,           # the quadtree depth of the leaves: 4**max_depth leaves, 1 to 5
    "band_threshold": 0.3,
    "max_weight": 8.0,
    "recurrence": {
        "network_model": "node_list",  # "grid" (EMR's model) by default
        "preset": "full_recurrent",    # or the classes with allow_hidden_to_hidden, as on the grid
        "iteration_level": 2,          # rounds of phase-2 discovery; HMR's presets carried 2
    },
}
```

What a generation computes (`emr_hyperneat/node_list.py`):

- Every leaf of the quadtree at `max_depth` is a candidate: HMR's active-leaf mask was all true, so
  no variance decided anything. `max_depth` keeps HMR's meaning, the depth of the leaves, so an HMR
  configuration carries over with the same number (the grid model's finest level lies one deeper).
- The CPPN is queried at (source, point) for every input and leaf as source and every output as
  target; the points are the leaves and their band neighbours, one ring outside [-1, 1]. Each
  weight is HMR's sparsification (threshold 0.2) and passes when its band, the larger of
  `min(top, bottom)` and `min(left, right)` of the differences to its four neighbours, exceeds
  `band_threshold`.
- Phase 1 connects the inputs to the leaves at or above them. Phase 2 explores from the leaves
  found, `iteration_level` rounds, along the configured connection classes. Phase 3 connects any
  leaf at or below an output to it. Cleaning keeps the connections that lie on a path from an input
  to an output.
- The forward updates every node at once, as HMR's did: `h_t = tanh(x W_in + h_(t-1) W_hh)` for
  `t = 1 .. T-1`, then `out = sigmoid(h_(T-1) W_out)`. The outputs read the hidden nodes one step
  behind, so `activate_time` must be at least 2. The steps follow the rule of the settling steps
  above (`full_recurrent`: 20, otherwise `2 (2**max_depth + 1)` unless set).
- The whole population is one compiled call on fixed shapes, three dense blocks per genome (inputs
  x leaves, leaves x leaves, leaves x outputs), where HMR looped over sources and genomes in Python.
  From its third generation on, a run compiles nothing new. `population_chunk_size` sets the
  genomes per compiled step (-1 a query budget, 0 the whole population); the result does not depend
  on it.
- The champion record, `evaluate_on_data` and `evaluate_best_all_modes` work as on the grid:
  `total_positions` counts the leaves and `total_connections` the three blocks. Each generation's
  metrics report `method` `node_list`, the hidden nodes after cleaning, the connections per block
  and `node_list_over_rn_cap`, the genomes HMR would have scored on its first 500 connections only.

There is no `recurrence.node_list` block and no connection cap: the leaves fix the shapes, and
depths above 5 are refused. Also refused under `network_model: node_list`, every reason in one
message:

- `variance_threshold`, `division_threshold`, `initial_depth` and `variance_rule` when set (there is
  no variance step), and `weight_threshold` (the node list sparsifies at HMR's 0.2);
- the grid's recurrence keys when set (`max_connections`, `max_sparse_conns`, the multi-hop keys,
  `hh_refresh_interval`, `hh_mask_change_threshold`, `hh_cache_trigger`, `use_dense_discovery`,
  `hh_discovery_mode`), `hh_cache_enabled: true` and `settling_rule: hmr`;
- `iteration_level` 0 with hidden-to-hidden connections, or above 0 without them; `activate_time`
  below 2; `max_depth` outside 1 to 5;
- per-node activation or aggregation (and so palettes and their strategies), hidden or output
  activations other than `tanh` and `sigmoid`, neuromodulation, multi-task evaluation, LEO,
  geometry seeding, `locality_radius`, swarm mode, and a CPPN other than 5 inputs and 1 output;
- streaming, depth offload, any `execution` key but `mode` (`standard` or `auto`), and the
  multi-device strategies, whether configured, passed to the constructor or assigned later (the
  generation and `run_until_threshold` check again);
- input or output coordinates on a leaf or on each other (the nodes would merge), and a substrate
  without them;
- a subclass that replaces the generation, `substrate_config`, `set_recurrence_config` and
  `run_generation_multitask`.

Six defects of HMR's variant are fixed, not reproduced. Its band scaled the neighbour weights by the
default `max_weight` 8 whatever the configuration said; here both ends use `max_weight`, which
equals HMR at 8. When no input or no output found a connection it invented one at weight 0.5, which
at its defaults gave 12 to 26% of genomes a network; here such a genome is empty, the constant 0.5
network. It evaluated each genome's first 500 connections in Python set order; here every
connection counts. From depth 6 its keys dropped every hidden connection; here depths above 5 are
refused. Its cleaning stopped after 50 iterations and kept a dangling middle of a longer chain; here
it runs to its fixpoint. Its sums ran in Python set order, so its fitness depended on hash order;
here the order is fixed. Compared on HMR's own populations, the networks are equal bit for bit on
CPU wherever none of these acts, and the fitness agrees within 1e-6 (at most 2.4e-7 measured).

To reproduce an HMR node-list run, rename its blocks to EMR's (`emrhyperneat`, `emr_hyperneat`),
set `network_model: node_list`, write HMR's connection classes out (HMR defaulted them to true, EMR
to false), give an HMR preset as EMR's preset with `iteration_level: 2`, and drop the keys HMR did
not read on that path (`max_connections`, `extra_randkey_split`, the variance settings, and the
substrate's `hidden_activation` and `output_activation`). An HMR preset replaced the whole
recurrence block; here the keys beside a preset override its values, as on the grid. HMR's
`enabled: false` or `preset: feedforward` ran HMR's feed-forward grid model, not a node list, so
its counterpart is EMR's default model.

HMR's GPU-resident variant of the same model ran a dense grid model at depth 4 and below and the
node list above it. Its dense model is EMR's default model with:

```python
"emr_hyperneat": {
    "variance_rule": "legacy",
    "extra_randkey_split": True,      # it split the key before ask
    "initial_depth": 0,               # its masks ignored it
    "band_threshold": 0.79734397,     # float32(tanh(0.1)) * max_weight, at max_weight 8
    "recurrence": {
        "enabled": True, "allow_hidden_to_hidden": True,
        "allow_backward": True, "allow_lateral": True, "allow_self_loops": True,
        "iteration_level": 1,
        "activate_time": 20,          # its dense path read the preset's count
        "max_sparse_conns": 1860496,  # at least P**2 (1364**2 at depth 4): it had no cap
    },
}
```

It kept a hidden-to-hidden weight where the raw CPPN output exceeded 0.1 in magnitude and ignored
`band_threshold`; `band_threshold = max_weight * tanh(0.1)` is the same test up to rounding at the
boundary. Its dense path took the steps from the preset's `activate_time`, which only
`full_recurrent` sets, and failed for the other presets. Above depth 4 it ran the node list, so its
counterpart there is the node-list model.

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
        "sparse_cap_policy": "truncate",   # 0.1.0 kept a genome's first pairs at the cap
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
| `static_gating` | Each hidden node's activity is multiplied by a gate in `[0, 1]`, derived from the node's own input weights: `tanh(2 * mean(abs(w)))`, then `gate_scaling` (`sigmoid` by default, with `gate_hardness` and `gate_threshold`). With `gate_source: cppn` (HMR's Level 1; `gate_from_cppn: true` is HMR's name) the gate is per connection, from a dedicated CPPN output, and scales the weights. 0.1.0 had the weight-derived gate only. |
| `context_gating` | A task context vector modulates the gates (XdG-style). With `gate_source: cppn` (HMR's Level 2) each case's connections are gated by `sigmoid(logit + context_influence * mean of the case's input statistics)` from a dedicated CPPN output, in a feed-forward pass, so hidden-to-hidden connections are refused there. |
| `modulatory_neurons` | Some of each genome's own active nodes set a per-case gain for the others: activity modulation, where Soltoggio's modulatory neurons gate plasticity, which EMR does not have. By default (`modulatory_source: weights`) they are `mod_neuron_ratio` (0.1, in (0, 1]; another value raises) of the genome's active positions, at least one, those of highest input-weight variance, and their mean activity, squashed by `tanh` and scaled by `mod_decay` (0.9), scales the nodes through `mod_connection_type` (`multiplicative`, `additive` or `gated`). `modulatory_source: cppn` (HMR's Level 3) takes them from a node-type CPPN output and their signal from a mod-factor output; the options below change one thing at a time between the two. In 0.1.0 every genome took the same positions, that fraction of every grid position. |
| `true_neuromodulation` | Neurotransmitter vectors `[DA, 5HT, NE, ACh]` + receptor densities. This is the mode the multi-task path uses; `num_nt_types` is usually 4. `receptor_from_weight: false` is HMR's Option B: each NT type's receptor density is a dedicated CPPN output at the node's self-connection, and the base gain comes from the weight output there (0.1.0 raised `NotImplementedError` at the first generation). |

Under `true_neuromodulation` the first three channels of the vector modulate a node through its
receptors, `m = receptors[:3] @ nt[:3]` with strength `s`, and `node_rule` (default `paper`) chooses
the node: `paper`, the published rule, applies the gain `clip(base_gain + s m, 0.1, 5)` and the bias
`s m` inside the activation and the gate `sigmoid(m)` after it; `hmr_full`, HMR FULL's rule, which
0.1.0's single-task path computed, applies the gain `base_gain + s m` with no bias and the gate
`sigmoid(5 m)`. `modulation_mode: gating_only` applies the gate alone (`sigmoid(s m)` under `paper`)
and `gain_bias_only` drops the gate. The fourth channel inverts the outputs when
`use_output_inversion` is set and the vector has four or more channels. `anti`, HMR ANTI's rule (its
anti-correlated-receptors variant), reads every channel, `m = receptors @ nt`, and applies the gain
`clip(base_gain + s m, 0.1, 5)` and the gate `sigmoid(m)`, with no bias and no output inversion; it
takes `modulation_mode: full` only and refuses multi-task branch gating (the routing would replace
its modulation over every channel). A rule other than `paper` raises outside `true_neuromodulation`.
The multi-task two-module mode computes the paper's rule
only and refuses the other two.

**Modulatory options.** `modulatory_source` (`weights`, the default, or `cppn`) is a bundle of five
options, and an option set in the block overrides its bundle's value:

| Option | `weights` | `cppn` | What it chooses |
|---|---|---|---|
| `modulatory_selection` | `variance` | `cppn` | Which nodes are modulatory: highest input-weight variance, or highest node-type output. |
| `modulatory_count_rule` | `floor` | `ceil` | How many: `max(1, floor(ratio x active count))` in float64, as since 0.2.0; `ceil(ratio x active count)`, exact; or `fixed`, `mod_neuron_count` per genome. |
| `modulatory_signal` | `mean_decay` | `sum_factor` | The signal: the modulatory neurons' mean activity, `tanh`, times `mod_decay`; or `tanh` of their activity times the mod-factor output, summed. |
| `modulatory_node_weights` | `variance` | `none` | Whether the signal is scaled per node by its input-weight variance. |
| `modulatory_scope` | `all` | `standard` | Which nodes it modulates: every node, or the standard ones only. |

`modulatory_cppn_read` (`input_mean`, HMR's read, or `self`) says where a CPPN output an option reads is
taken, `modulatory_cppn_outputs: always` keeps both CPPN outputs even when no option reads them (`read`,
the default, carries the ones read), and `modulatory_matched_gain` (`none`, `mean` or `shift`), a
control, replaces each case's signal by its mean over the cases or by the previous case's. A setting
that no option reads raises, as does an option outside mode `modulatory_neurons`.

A `preset` (`static_gating`, `xdg_style`, `modulatory_only`, `true_neuromodulation`,
`true_neuromodulation_4nt`, `true_neuromodulation_5nt`, or `disabled`) sets the mode of its level
(`xdg_style` is `context_gating`) and its parameters, which the block's own keys override. In 0.1.0
a preset left the mode `disabled`, so a run configured by a preset alone had no modulation. A preset
that combines levels (`full`), an unknown preset or mode, a `mode` that contradicts the preset,
`branch_gating: true` (never implemented; the multi-task routing is `multitask.branch_gating_mode`),
and `enabled` that contradicts the mode (true with no mode or preset, false with one) raise. The
block's other keys raise too: the level flags (`static_gating: true` and the others; one mode is one
level) and `context_derivation` (never read; context gating takes the current task index) were
ignored in 0.1.0. `gate_from_cppn: true` is HMR's name of `gate_source: cppn` and raises outside the
two gating modes. `context_source` accepts only `'task_id'`, its default, which says what context
gating reads; `'input'` and `'learned'` raise (0.1.0 ignored them).

**HMR's mode names.** HMR's neuromodulation variant used preset names as modes, and they translate
to EMR's blocks: the Option A names (`true_neuromodulation_4nt_option_a_tanh` and the others of
`HMR_NEUROMODULATION_MODES`) are `true_neuromodulation` with the name's NT count and receptor
derivation, read at the self-connection as HMR read them; `true_neuromodulation_4nt` is Option B with
four NT types; `xdg_style` is `context_gating` and `modulatory_only` is `modulatory_neurons`, both
from the CPPN. `custom` with HMR's level flags is the level HMR evaluated, the highest of true
neuromodulation, modulatory neurons, context gating and static gating: `true_neuromodulation` alone is
Option A or B as its flags say; static or context gating alone is that level with a gate from a
dedicated CPPN output (`gate_source: cppn`); modulatory neurons alone take theirs from the CPPN
(`modulatory_source: cppn`); static with context gating is context gating, as `xdg_style` is. Another
combination raises, naming the level HMR evaluated (its lower levels made the CPPN evolve outputs
nothing read), and so does `custom` with no level. A key that contradicts the name raises, as does
`full` (HMR evaluated Level 3 only while its CPPN evolved a gate output nothing read). Two of HMR's
names are EMR's own and keep EMR's meaning: `true_neuromodulation` and `static_gating`. 0.1.0
accepted HMR's mode names and failed at the first generation, and ran HMR's presets without
modulation.

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
Three keys are accepted only at their default: `orthogonality_bonus` and `specialization_bonus`
(default `0.0`; each adds one constant to every genome, which never changes a rank) and
`confusion_eval_frequency` (default `0`, read by nothing). The per-genome bonuses are HMR's, with
its defects fixed: `orthogonality_bonus_weight` adds, per genome, how orthogonal its task
fingerprints are (its hidden activity averaged over each task's cases;
`orthogonality_metric`: `cosine_mean`, `cosine_max` or `correlation`), and
`specialization_bonus_weight` its mean fitness under each task's own NT vector minus its fitness on
the same data under another task's vector (with two tasks, the other one's; it needs two or more
tasks). `branch_gating_mode` (`spatial`, by quadrant of the grid, or `hierarchical`, by height) gives
each region of the grid one receptor channel, in the feed-forward and the recurrent evaluation alike;
it needs three or more NT types and refuses `two_module_mode`. `modulation_penalty`, a bonus despite
its name, adds the penalty times the variance over tasks of a genome's mean modulation over its
active positions; it needs two or more tasks. 0.1.0 accepted both at any value and applied neither.
`run_generation_multitask(state, MultiTaskConfig(tasks={name: (inputs, targets)}, aggregation=...,
weights=..., nt_vectors=...))`, HMR's call form, runs the same generation and returns
`MultiTaskMetrics` keyed by task name; `evaluate_with_neurotransmitter` takes a task name in place
of the vector for that task's preset NT vector. `run_generation_multitask`
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
the incoming weights; `modular` takes `dynamic_functions.modular.aggregation_method` (`none`, the
default, for `sum`, or one of the four interpretations). `aggregation.palette` (indices or names of
`AGGREGATION_LIST`), or `agg_palette` set on the algorithm object, restricts the per-node modes to its
functions as the activation palette does: the decode picks a palette position and maps it to the
function's index, and evaluation covers the whole list. A palette entry outside the list raises,
checked every generation, since a runner may set `agg_palette` between generations. 0.1.0 read
neither, so the aggregation palettes that the bio-inspired palette paper's dual strategies compute
(`papers/emr-dynamic-functions-bio-inspired/scripts/runners/`) reached no evaluation; a palette under
a mode that decodes no per-node aggregation raises. **Changed in 0.2.0:** the key
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

## `leo` (link expression output)

LEO (Verbancsics and Stanley 2011) gives the CPPN a second output that decides whether a connection
exists, apart from its weight:

```python
"emr_hyperneat": {"leo": {"enabled": True, "threshold": 0.2, "mode": "compound"}}
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `False` | Adds LEO's output, CPPN channel 1 (`LEO_CHANNEL`); the per-node function channels follow it. |
| `threshold` | `0.2` | The LEO value (after `tanh`) above which a connection is expressed. |
| `mode` | `compound` | `compound`: a connection needs its weight above the weight threshold and LEO above `threshold`. `replacement`: LEO alone decides, and the weight is used unthresholded. `hmr_compound`: HMR's LEO gate, the weight scaled to `W = tanh(w) * max_weight` and kept where `abs(W)` exceeds the weight threshold and LEO exceeds `threshold`, with no rescaling of the kept range; refused with `locality_radius` (HMR penalized the scaled weight, EMR's LEO paths the tanh-normalized one). |
| `output_index` | `1` | Must be `1`: LEO is read from that channel on every path. |

Input-to-hidden, hidden-to-output and hidden-to-hidden connections are gated by the same rule in the
standard and streaming modes. Depth offload and the multi-device pipelines, which build their own
weights, refuse LEO, and so does the substrate API. Under the multi-hop expansion a direct
connection that LEO's replacement mode expressed below the weight band stays a connection. HMR's LEO
variant returned a population without the LEO output, so its runs evaluated empty networks; a run
with `hmr_compound` computes HMR's gate on networks that have one, and does not reproduce those
numbers.

## Continual learning

Methods of `EMRHyperNEAT`, from HMR's dynamic-functions variant, with its defects fixed (the
champion read from the right population, injection that spares the champion's slot, a freeze its
tests can see, an evaluator that reproduces the training fitness):

| Method | What it does |
|---|---|
| `get_best_genome_with_fitness()` | The last evaluated generation's champion, as the generation recorded it. |
| `get_best_genome(state)` | The same champion's `(nodes, conns)`; `state` is not read. 0.1.0 returned the whole population's node array. |
| `inject_genome_into_population(state, nodes, conns, num_copies=1, slots=None)` | Puts copies of a genome into the population the next generation evaluates. |
| `freeze_topology(state)` / `unfreeze_topology(state)` / `is_topology_frozen(state)` | Stops and restores structural mutation (node and connection additions and deletions). |
| `evaluate_genome_on_problem(state, nodes, conns, problem)` | A genome's fitness on a problem's data (or `inputs` and `targets`), as the generation loop would score it. |
| `create_state_for_new_problem(problem, seed=42, inherit_genome=None, num_copies=1)` | A fresh population for another problem, optionally seeded with copies of a genome. |

## `geometry_seeding` (shipped, not part of any published paper)

A locality bias on connection weights, after the geometry-seeding idea in Risi & Stanley (2012):
enabling it switches the CPPN from its 5-input form to a 7-input form that also receives the
source->target coordinate offsets, so weight patterns can condition on relative geometry. It is
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
| `multi_gpu_strategy` | unset (`single_gpu`) | Case-insensitive string selecting the multi-GPU strategy: `"single_gpu"`, `"full_pipeline_parallel"`, `"streaming"`, `"eval_only_parallel"`, `"population_parallel_process"`, the split routes `"population_pmap"`, `"position_sharding"` and `"hybrid"`, or the island model `"island_model"`; the split routes read the `parallel` block and the island model the `islands` block ([below](#multi-device-routes-parallel-and-islands)). Thirteen older alias names are accepted and mapped onto those nine, here and in the constructor's `strategy` argument alike; an unknown value raises (0.1.0 warned and kept the default). `population_parallel_process` is not recommended: it is the least exercised path, and in development runs at population 1000 every worker ran out of GPU memory; a worker that fails raises. `eval_only_parallel` and `population_parallel_process` evaluate `tanh` nodes with sum aggregation and no neuromodulation, so they refuse dynamic functions, aggregation and neuromodulation. What each one parallelizes, and the fact that it engages only through `run_until_threshold`: [architecture](architecture.md#what-the-multi-gpu-strategies-parallelize). |
| `extra_randkey_split` | `False` | `True` reproduces the per-generation PRNG split of the HMR engine (release 0.1.0 ships it), for bit-exact replication of the results recorded on it. Details: `papers/emr-dynamic-functions/REPRODUCIBILITY.md`. |
| `locality_radius` | unset | Scales each input-to-hidden and hidden-to-output weight by `min(1, radius / distance)` between its endpoints, so distant connections are weakened. |
| `fitness_rescale_variance` | unset | Scores `max(0, 1 - MSE / variance)` instead of `1 - MSE`, which spreads the fitness of targets with a small variance; must be positive (0.1.0 ignored a value that is not). |

Depth offload, `population_parallel_process`, `eval_only_parallel` and the `full_pipeline_parallel`
pipelines score `1 - MSE` on weights they build themselves, so they refuse a `fitness_mode` other
than `mse`, `locality_radius`, `fitness_rescale_variance` (0.1.0 ignored them there), LEO and
`recurrence.settling_rule: hmr`.

## Multi-device routes: `parallel` and `islands`

The split routes and the island model run the standard generation, so they take every setting it
takes. Each block is read only under its strategy and raises under another. What the routes split
and replicate: [architecture](architecture.md#what-the-multi-gpu-strategies-parallelize).

### `parallel`: the split routes' device grid

`population_pmap`, `position_sharding` and `hybrid` run `single_gpu`'s generation and spread two of
its steps over a grid of `population_devices x position_devices` devices: the one-output CPPN query,
by genomes, by grid positions or both, in chunks of at most 100 genomes per device as on one device,
and the evaluation, by genomes over every device of the grid. The rest of the generation runs on the
grid's first device, where the results are gathered: the masks, the hidden-to-hidden discovery at
any iteration level, compaction, the caches and the records. On CPU devices a run equals
`single_gpu`'s bit for bit; on GPUs it follows it to float32 rounding.

| Key | Default | Meaning |
|---|---|---|
| `population_devices` | every visible device, under `population_pmap` | The devices the genomes are split over. `position_sharding` refuses a count above 1. |
| `position_devices` | every visible device, under `position_sharding` | The devices the CPPN query's grid positions are split over. `population_pmap` refuses a count above 1. |

- `hybrid` needs both counts. A count is a positive integer, and the grid must fit the visible
  devices. With one visible device, a route runs the one-device loop and records `single_gpu`.
- Refused at the run: an explicit `execution.mode` of `streaming` or `depth_offload` and the legacy
  `enable_streaming` (the routes spread the standard generation); `auto` when it resolves to depth
  offload; `execution.generation_loop: while_loop`; a population that leaves a device fewer than two
  genomes, since XLA evaluates a batch of one otherwise, by a float32 ulp. The node-list model
  refuses the routes at configuration.
- LEO, a multi-output CPPN (per-node functions read from the CPPN, palette evolution) and the
  self-connection query keep their CPPN query on the first device, so only their evaluation is
  spread.
- Each device holds one chunk's queries and its range of genomes with a copy of the data. The first
  device also holds what one device holds outside the query and the evaluation (the masks, the
  hidden-to-hidden discovery, the weight matrices), so a configuration whose generation does not fit
  one device there does not fit a split route either.
- The routes record the champion, so `evaluate_on_data` and `get_best_genome_with_fitness` work
  after them. The result names the route and its device count; `get_num_devices` gives the grid's.

### `islands`: the island model

`island_model` splits the population into islands that evolve apart, generation by generation in
lockstep, and exchange their best genomes every `migration_interval` generations.

```yaml
emr_hyperneat:
  multi_gpu_strategy: island_model
  islands:
    count: 3                    # required, at least 2
    populations: [16, 24, 20]   # optional, summing to population_size (60 here); default: equal
    devices: [0, 1, 1]          # optional; default: island i on device i mod the visible devices
    migration_interval: 2       # required; 0: no migration
    migrants: 1                 # required with migration
    topology: ring              # ring (default), star or random
```

| Key | Default | Meaning |
|---|---|---|
| `count` | required | The number of islands, at least 2. |
| `populations` | `population_size` split equally | One population per island, summing to `population_size`. The equal split needs `population_size` divisible by `count`; otherwise the error message gives a near-equal list to paste. Each island needs more genomes than `neat_selection.genome_elitism` (15 by default): TensorNEAT copies that many unmutated, so a smaller island only copies itself. |
| `devices` | island i on device i mod the visible devices | One visible device per island. Placement changes where the arrays live, never a result. |
| `migration_interval` | required | Migrate after every this many generations; `0` runs the islands without migration. |
| `migrants` | required with migration | The genomes each island sends per migration. One migration's immigrants (`migrants`, or `migrants x (count - 1)` with `star`) must fit below each receiver's best `genome_elitism` genomes, which are never replaced. |
| `topology` | `ring` | `ring`: island i sends to island i + 1; `star`: every island to every other; `random`: each island to one other, drawn from the run's seed and the migration. |

- **An island is a `single_gpu` run** at its own population, initialized with the seed
  `island_seed(seed, i)` (`emr_hyperneat.emrhyperneat`): the run's seed for island 0, and for island
  i a seed drawn from `(seed, i)` by numpy's `SeedSequence`. The HMR engine's `seed + i` made island
  1 of seed 42 island 0 of seed 43. The state passed to `run_until_threshold` is not evolved: each
  island starts from its own.
- **Lockstep.** Every island runs a generation, in index order and from one thread, before any runs
  the next; the run stops at the end of the generation in which an island reaches the target.
- **Migration.** After generation M, a multiple of `migration_interval`, each island sends copies of
  its best `migrants` genomes of that generation, with the fitness they earned. At generation M + 1
  they replace the receiver's worst genomes in its selection step, so they compete with that
  fitness, and their hidden node keys move above the receiver's largest, so crossover never takes
  them for unrelated nodes. Each island's records keep its own evaluation, and its random keys are
  those of the run without migration.
- **Records.** The result lists each island's seed, population, device, state, best fitness and
  history, and each migration. Its `state` and champion are those of the best island at the last
  generation, so `evaluate_on_data` and `get_best_genome_with_fitness` work after the route.
  `get_num_devices` and the result's `devices` count the devices that host islands.
- **Refused:** an execution mode other than `standard` and `execution.generation_loop: while_loop`,
  at the run. With migration, also `recurrence.hh_cache_enabled` (it reuses connections by
  population slot, so an immigrant would be scored with the connections of the genome it replaces)
  and palette evolution or a palette strategy (each island's palette would decode an immigrant's
  per-node functions otherwise than its sender's did).

## Execution mode

Controls **where a generation's intermediate arrays live**, not what is computed. All three modes
produce the same results (to float32 rounding where compaction rounds differently, on x86 CPUs and
GPUs) for the configurations each accepts, under either `variance_rule`; they differ in peak
memory. (In 0.1.0 streaming and depth_offload ignored `variance_rule: legacy`
and ran the canonical rule.) This is what decides whether a given depth runs at all.

In 0.1.0 `depth_offload` split the random key before every selection step, which the other
modes do only under `extra_randkey_split`, so from generation 1 a default depth-offload run bred
another population than a standard run of the same seed. 0.2.0 draws the same keys in every
mode. `extra_randkey_split: true` draws 0.1.0's keys there: a feed-forward depth-offload run then
reproduces 0.1.0's bit for bit on a CPU, and a recurrent one does with the settings of
[Reproducing a 0.1.0 recurrent run](#reproducing-a-010-recurrent-run).

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
| `standard` (default) | All on device | Depth 7 and below. Highest peak memory. This is the path the published results and the golden tests were produced with. |
| `streaming` | One hierarchical level at a time | When the CPPN queries of all positions do not fit at once; W1 and W2 stay at full size. Only the CPPN queries are streamed: the evaluation is the standard one, and a recurrent evaluation is not chunked over samples. |
| `depth_offload` | Assembled in host RAM (or a memmap), moved to the device one chunk at a time | Depth 11+, where the matrices exceed device memory outright. Depth 11 is 22.4M candidate positions, about 380 MB per genome. Evaluates `tanh` nodes with sum aggregation and no neuromodulation, so it refuses dynamic functions, aggregation and neuromodulation (0.1.0 evaluated without them), and it records no champion for `evaluate_on_data`. Its population-chunked loop (a population above `max_population_chunk`, or weight matrices above `target_cpu_gb`) refuses `recurrence.hh_cache_enabled`: that loop never consulted the cache, so with the cache on it rediscovered the hidden-to-hidden connections every generation. It is the one mode that chunks a recurrent evaluation over the samples as well as the population (`evaluate_population_recurrent_double_chunked`, sized by `recurrent_target_memory_gb`); `auto` selects it from `depth_threshold` up, and on a GPU when the standard path would not fit. |
| `auto` | - | Picks by `max_depth` against the two thresholds. On a GPU backend the pick is also memory-aware: a `standard` selection escalates to `depth_offload` when the standard path's calibrated working-set estimate exceeds the device. Population size can push `standard` past device memory at depths the thresholds alone would allow. Inert on CPU; never overrides an explicit mode. A configuration `depth_offload` cannot evaluate is refused when `auto` selects it, at `create_config` or, for the memory-aware escalation, at the first generation; set `streaming` explicitly for it. |

**Omitting the `execution` block gives `standard`.** `enable_streaming: true` is a valid spelling
of `mode: streaming`.

### Memory budgets

Defaults are usually right; reach for these when a run OOMs or when you want it to use less.

| Key | Default | Meaning |
|---|---|---|
| `target_cpu_gb` | `32.0` | Host RAM budget for the weight matrices. Sets the population chunk size. |
| `max_population_chunk` | `None` | Override the computed genomes-per-chunk. |
| `target_shard_gb` | `0.5` | Device memory per position shard during CPPN queries. Sized on the CPPN forward's while-loop carry (2 x `max_nodes` floats per genome-position), the measured dominant term. |
| `max_lookahead_positions` | `5000000` | Memory budget for the variance lookahead. Levels larger than it are computed in parent-row slabs: bounded memory, identical substrate. |
| `skip_dead_levels` | `True` | Skip weight processing for levels with zero reached positions. Changes cost, never results (asserted per pair). Default `True`; the reported runtimes were measured with it off, so the baseline capture pins `False`; the GECCO from-scratch runners run the standard path, where the key does not act. Acts on the `depth_offload` path only. |
| `target_eval_shard_gb` | `1.0` | Device memory per shard during evaluation. |
| `cppn_pos_chunk_size` | `100000` | Upper clamp on positions per CPPN query. |
| `eval_chunk_levels` | `3` | Levels loaded at once during evaluation. |
| `eval_chunk_positions` | `1000000` | Maximum positions per device chunk. |
| `recurrent_target_memory_gb` | `2.0` | Peak budget for the recurrent evaluation of `depth_offload`, which chunks population *and* samples. `standard` and `streaming` do not chunk samples and do not read it. |
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
`-1` (the default) is AUTO: a budget gate that returns the full population unchanged whenever
the largest level's query carry fits `target_shard_gb`, and chunks only where the unchunked query
would OOM. Per-genome query results are independent, so chunking never changes values.

### The generation loop

`execution.generation_loop` (default `"python"`) sets how `run_until_threshold` iterates its
generations on one device:

- `"python"`: one generation call per generation; every feature.
- `"while_loop"`: the HMR engine's GPU loop (release 0.1.0) over the plain feed-forward
  generation, compiled once inside `jax.lax.while_loop`, with the results read back once. The
  compiled loop is kept per `max_generations` until `create_config`, `initialize` or a topology
  freeze, so a second call with the same shapes compiles nothing.

It changes speed only, up to rounding. On a CPU the loop equals the same generation compiled one
call at a time, bit for bit; its fitness differs from `"python"`'s by at most one float32 ulp
(XLA fuses the compiled evaluation), and the generation count, the stop generation and the final
random key are `"python"`'s. The stop rule is the Python loop's comparison with the target. Its
speed is not measured.

The loop applies `eval_chunk_size`, `fitness_mode`, `fitness_rescale_variance`,
`locality_radius`, geometry seeding, `variance_rule`, `initial_depth` and `extra_randkey_split`.
It refuses, naming each setting: hidden-to-hidden connections and their cache, the node-list
model, per-node activation and aggregation modes other than `disabled` and `global`, LEO and
multi-output CPPNs, neuromodulation and multi-task, streaming and depth offload, palette evolution
and the live palette strategy, and the multi-device routes on two or more devices. `verbose` warns
that nothing prints per generation.

Its result has the one-device route's keys, with `sparse_hh_stats` None, `nan_fitness_count`
(the NaN fitness values of the run, told as -inf) and `generation_loop`. It keeps no
per-generation record: `evaluate_on_data` raises after it, and the per-generation mean, species,
topology and step timings need `"python"`.

The HMR engine's defects are fixed rather than reproduced: its per-node variants evaluated plain
tanh/sigmoid networks in this loop (here those settings are refused); it ignored the Python path's
memory settings (here the loop runs the default path's own generation); it compiled again on
every call (here the loop is cached); it compared the best with the target cast to float32, which
stops a generation early when the two meet below the target; and a NaN CPPN output at an inactive
position made a genome NaN (here the weights are masked with `jnp.where`). Against the HMR
engine's own loop (its level-0 rule and key split, depth 2, population 60) it gives the same
generations and final key, and the same best of every generation to 1e-6
(`emr_hyperneat/tests/test_generation_loop.py`).

## 64-bit floats

EMR computes in float32 unless JAX's x64 mode is on (`JAX_ENABLE_X64=1` in the environment, or
`jax.config.update('jax_enable_x64', True)` before the first JAX computation). Under x64 the
fitness and the species bookkeeping are float64 (the problem's inputs stay float32), so results
differ from float32 runs; no published result ran under x64. In 0.1.0 x64 stopped every run at
its first generation: TensorNEAT's species update traced two `lax.cond` calls whose branches differ
in dtype under x64, and raised "cond branches must have equal output types". `create_config` now
gives TensorNEAT's species module a `lax.cond` that casts both branches to their promoted dtype
(`emr_hyperneat/x64_compat.py`); under float32 nothing changes.

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
reproduce a 0.1.0 run that set them. Any other key in the `neat` block is not read: it raises (an
operator setting placed there names its block), except `species_size`, which the per-node paper's
runners carry and which is accepted.

## Settings that raise instead of being ignored

These settings used to be accepted and then ignored, so a run did something other than its
configuration said. Each now raises a `ValueError` that names the setting and what to do instead:

- **A level flag against an HMR mode name.** HMR merged a block's `static_gating`, `context_gating`,
  `modulatory_neurons` and `true_neuromodulation` into the preset its mode named and evaluated the
  highest level set, so `xdg_style` with `modulatory_neurons: true` ran modulatory neurons, while
  the translation dropped the flag and ran context gating. A flag whose value differs from the
  named preset's raises (use mode `custom`); a flag that repeats it is accepted.
- **A connection class without hidden-to-hidden connections.** With `enabled: true` and
  `allow_hidden_to_hidden: false` the network is feed-forward, and `allow_backward`,
  `allow_lateral` or `allow_self_loops` was kept and ignored.
- **`dynamic_functions.modular.aggregation_method` without `aggregation.mode: modular`**, the only
  setting that reads it.
- **`run_generation_multitask`'s other call forms.** On a configuration object set by hand, or by
  HMR's `MultiTaskConfig` call form, `create_config`'s checks never ran: a constant
  `orthogonality_bonus` or `specialization_bonus` above 0, a `confusion_eval_frequency` other than 0
  (HMR's default was 1 and nothing ever read it) and an unknown `generalist_bonus_type` were
  ignored. Per-node functions or aggregation are refused on every call, not only with
  `multitask.enabled`.

A non-sigmoid output activation on the plain path with `aggregation.mode: modular` and
`aggregation_method: none` still raises, as before: its message names the setting, and the remedy
(`dynamic_functions.mode: global` with a tanh hidden activation) runs.

## Keys that parse but do not act

The keys below are accepted and change nothing, or accepted only at the value that changes nothing;
they are listed so silence is never mistaken for effect. Every other key the engine does not read
raises, in `emr_hyperneat` and its blocks and at the levels outside them
([above](#configuration-reference)).
When in doubt, copy a config from a paper runner under `papers/*/scripts/`. The EMR runners among
them use this format.

| Key | Where it sits | Status |
|---|---|---|
| `species_size` | `neat` | Not read, and accepted without a warning; the NEAT operators read `neat_mutation`, `neat_species` and `neat_selection` ([above](#neat--population)). |
| `critical_periods` (`phase1_end`, `phase2_end`, `min_plasticity`), `enabled: false` | `dynamic_functions` | Parsed and stored, read by nothing; `enabled: true` raises. The critical periods of the palette paper are part of its strategies (`papers/emr-dynamic-functions-bio-inspired/scripts/palette_strategies/`). |
| `division_threshold` | `emr_hyperneat` | Warns (`DeprecationWarning`) and names the live controls. |
| `iteration_level: 1` | `emr_hyperneat` | The multi-hop level is `recurrence.iteration_level`; another value raises. |
| `sparse_forward_threshold`, `fuse_w1_computation`, `skip_unused_masks` | `emr_hyperneat` | HMR's base-engine settings: checked, and no value changes a result ([above](#emr_hyperneat-substrate-discovery)). |
| `seed_weight` | `geometry_seeding` | Read by nothing ([above](#geometry_seeding-shipped-not-part-of-any-published-paper)). |
| `use_vectorized_discovery` | `recurrence` | Retired: it was stored and never consulted. |
| `max_connections` | `recurrence` | Checked (100 or more) and never read; it warns (`UserWarning`). The cap on each genome's hidden-to-hidden connections is `max_sparse_conns`: a run that reproduces an HMR unified-model run sets it at least to the run's largest set, since a genome over it raises. Refused under `network_model: node_list` ([above](#hmrs-node-list-model)). |
| `orthogonality_bonus`, `specialization_bonus`, `confusion_eval_frequency` | `multitask` | Accepted at their defaults only ([Feature 3](#feature-3-neuromodulation-multi-task)). |

Next: [reproducing the paper experiments](reproducing-experiments.md), or back to
[writing experiments](writing-experiments.md).
