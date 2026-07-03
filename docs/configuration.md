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

You can omit almost everything and rely on defaults; the sections below give the knobs that matter
and sensible values (the ones the papers use).

> **A note on defaults.** When you omit a key, the runtime parser fills in its own default, which
> sometimes differs from the values the test factories pass explicitly. The values below are the
> *recommended* ones. Set them explicitly and you will not be surprised.

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

| Key | Recommended | Meaning |
|---|---|---|
| `initial_depth` | `0` | Starting subdivision depth (rarely changed; leave at 0). |
| `max_depth` | `2`–`3` | Maximum grid depth. Candidate positions grow ~4× per level (depth 2 → 84, 3 → 340, 4 → 1364). Higher depth = larger substrate = slower. |
| `variance_threshold` | `0.03` | Minimum CPPN-output variance for a position to survive the mask. Lower = denser substrate. |
| `division_threshold` | `0.5` | Region-subdivision threshold during discovery (internal, leave at default). |
| `band_threshold` | `0.3` | Band-pruning threshold for connection expression (internal, leave at default). |
| `max_weight` | `8.0` | Scales CPPN output to a connection weight (internal, leave at default). |
| `verbose` | `False` | Print substrate-build logging. |

`division_threshold`, `band_threshold`, and `max_weight` are substrate-discovery internals. The
examples omit them and use these defaults; change `max_depth` and `variance_threshold` first.

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
| `cppn_output` | The CPPN emits a per-node activation index from a palette. Set `num_activations` (e.g. 4 or 6) or a `palette` (below). |
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

`iteration_level` (default 2) sets how many propagation hops; `hh_cache_enabled` (default `True`)
caches the discovered h→h connections between generations.

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

For multi-task, the companion `multitask` sub-dict (`enabled`, `num_tasks`, `task_names`,
`fitness_aggregation` ∈ `mean`/`min`/`weighted`/`product`/`softmin`/`harmonic`) plus the
per-task NT vectors drive `run_generation_multitask`. The runners under
`papers/emr-neuromodulation/scripts/runners/` are the reference.

## Feature 4: `aggregation` (partial)

Per-node aggregation (how a node combines its inputs) is present in the code but **only exercised as
part of the bio-inspired palette meta-learning**, not as a standalone knob. In particular, the
standalone "global aggregation" presets do not change behavior through a single config key, so they
are not documented as user controls here. If you need evolved aggregation, follow the bio-inspired
palette runners under `papers/emr-dynamic-functions-bio-inspired/scripts/runners/`, which co-evolve
activation *and* aggregation palettes. The aggregation functions themselves are
`sum, mean, max, min, product, maxabs`.

## `neat` / population

In the EMR path the `neat` block reads only the population size:

```python
"neat": {"pop_size": 150}     # or set top-level "population_size"
```

The remaining NEAT hyperparameters (mutation rates, species/compatibility thresholds, elitism) are
fixed inside the engine and are not read from this block, so, for example, `species_size` placed
here has no effect. If you need to change those, you are into engine territory, not configuration.

## A note on partial / experimental knobs

This is research code. A few configuration keys exist but are not wired end-to-end (the aggregation
"global" presets above; `species_size` in the `neat` block; `palette_evolution.enabled`, which is
handled by the bio-inspired meta-learning rather than this parser). The docs deliberately point you
at the paths the papers and the test suite exercise. When in doubt, copy a config from a paper runner
under `papers/*/scripts/`. Those are known-good.

Next: [reproducing the paper experiments](reproducing-experiments.md), or back to
[writing experiments](writing-experiments.md).
