# Writing your own experiments

This page shows how to use EMR-HyperNEAT from your own code: define a problem, configure the
substrate, evolve, and read the result. For every configuration option see the
[configuration reference](configuration.md).

## The three objects you work with

1. **A problem**, a plain Python object that exposes your data and shapes (no base class to inherit).
2. **A config dict** that describes the substrate and which features to switch on.
3. **`EMRHyperNEAT`**, the algorithm; you call three methods on it.

## The public API

```python
from emr_hyperneat import EMRHyperNEAT

algo  = EMRHyperNEAT()                       # no constructor arguments
cfg   = algo.create_config(config)           # parse + validate your config dict
state = algo.initialize(cfg, problem, seed=42)   # build the substrate + JIT-compile

# one generation: returns the new state and a metrics object
state, metrics = algo.run_generation(state, problem, verbose=False)
print(metrics.best_fitness)
```

| Method | Signature | Returns |
|---|---|---|
| `create_config` | `create_config(params: dict)` | an opaque config object. Pass it straight to `initialize`, don't introspect it |
| `initialize` | `initialize(config, problem, seed=42)` | the initial `state` you thread through the loop |
| `run_generation` | `run_generation(state, problem, verbose=True)` | `(new_state, metrics)`. Feed `new_state` into the next call |
| `run_until_threshold` | `run_until_threshold(state, problem, target_fitness, max_generations, collect_history=False)` | a dict with `generations`, `best_fitness`, `state`, `history` (if requested), and the run's `route` (the strategy route it took), `devices` (the devices it used) and `execution_mode` |
| `run_generation_multitask` | `run_generation_multitask(state, problems, neurotransmitters=None, aggregation_method='mean')` | `(new_state, metrics)`. Requires neuromodulation in `true_neuromodulation` mode |
| `population_substrates` | `population_substrates(state)` | the substrates of the population, as the generation loop builds them: `W1`, `W2`, the hidden-node `mask`, per-node function indices, and the hidden-to-hidden connections under per-genome discovery. Run them with `substrate_forward(substrate, inputs)` |
| `substrate_config` | `substrate_config()` | the `SubstrateConfig` the loop uses, for building substrates from CPPN outputs of your own (`substrate_queries`, `build_substrate`, `hh_query_rows`, `build_recurrent_substrate`). Raises for configurations outside the standard path the API describes |

### The metrics object

`run_generation` returns an `AlgorithmMetrics` with at least: `generation`, `best_fitness`,
`mean_fitness`, `min_fitness`, `max_fitness`, `std_fitness` (the last three over the finite
fitness values), `evaluations`, `time_elapsed`, the evaluated population's CPPN species as
TensorNEAT assigned them (`num_species`, `species_sizes`, `species_fitness`, the best fitness of
each), its topology (`hidden_nodes_mean` / `_std`, the active positions, and `connections_mean` /
`_std`, the nonzero input and output weights plus the h->h connections), and a `custom_metrics`
dict. For EMR the useful `custom_metrics` keys are `total_positions` (the candidate grid size),
`num_levels` and `variance_threshold`, `avg_hidden_nodes` / `max_hidden_nodes` (how many survived
variance masking), `position_utilization` (the sparse fraction), and `nan_fitness_count`.
**Changed in 0.2.0:** 0.1.0 reported no species on the standard path, one species holding the whole
population on the multi-task and depth-offload paths, and no topology.

Those three keys are also your substrate diagnostics — they feed straight back into the two knobs
that shape the substrate: if `position_utilization` sits near zero and `avg_hidden_nodes` is tiny,
the mask is starving the network — lower `variance_threshold` (or force floors with
`initial_depth`). If `avg_hidden_nodes` stops growing as you raise `max_depth`, added depth is
ceiling the search never uses — deeper is only slower. A healthy run keeps a small active set
(utilization well under 1%) that still moves across generations.

### Which run method

| You want | Use | Notes |
|---|---|---|
| Your own loop, one generation at a time | `run_generation` | Returns `(state, metrics)`. Single-device by design; "solved" is your own check against `metrics.best_fitness`. |
| The algorithm to own the loop | `run_until_threshold` | Takes `target_fitness` + `max_generations` and stops itself. Returns a **dict** (`generations`, `best_fitness`, `state`, and `history` if requested), not the tuple. **This is also the only entry point that engages `multi_gpu_strategy`** — a loop over `run_generation` runs on one device no matter what the config says. |
| Several tasks on one substrate | `run_generation_multitask` | One generation across a list of problems (see [Multi-task](#multi-task-one-substrate-several-tasks)). Requires `true_neuromodulation`. |

## The Problem interface

A problem is duck-typed. Give it these attributes and one method:

```python
class MyProblem:
    input_shape  = (3,)     # (number of inputs,) INCLUDING the bias column if you use one
    output_shape = (1,)     # (number of outputs,)
    jitable = True          # your get_data() returns plain numbers, safe to JIT
    use_bias = True         # whether your inputs carry an explicit bias column
    fitness_threshold = 0.95   # what you consider "solved" (your loop decides when to stop)
    is_exhaustive_dataset = True   # True for combinatorial tasks (XOR, parity, logic gates)

    def get_data(self):
        # return a list of (input, target) pairs
        return [([0., 0., 1.], [0.]), ([0., 1., 1.], [1.]),
                ([1., 0., 1.], [1.]), ([1., 1., 1.], [0.])]
```

- `get_data()` returns a **list of `(input, target)` pairs**. Each `input` has length
  `input_shape[0]`, each `target` has length `output_shape[0]`. The pairs are stacked into arrays of
  shape `(num_cases, num_inputs)` and `(num_cases, num_outputs)`.
- If you use a bias, include an explicit `1.0` column in every input and count it in `input_shape`.
- `is_exhaustive_dataset` matters only under multi-GPU: when `True`, data-sharded parallelism
  (`full_pipeline_parallel`) is automatically disabled, because a combinatorial task's fitness is
  only meaningful over **all** samples together — splitting XOR's four rows across devices would
  score each genome on a fragment of the truth table. Statistical datasets can leave it `False`
  (the default when the attribute is absent).

### Substrate coordinates come from the config, not the problem

EMR-HyperNEAT places one input node and one output node per **coordinate** you list in the config.
The contract you must satisfy:

```
len(substrate.input_coords)  == input_shape[0]      (bias included)
len(substrate.output_coords) == output_shape[0]
```

Each coordinate is a 2D `(x, y)` position on the substrate plane. A common convention is to spread
inputs along the bottom edge and outputs along the top:

```python
n_in = problem.input_shape[0]
input_coords  = [(-1.0 + 2.0 * i / (n_in - 1), -1.0) for i in range(n_in)]  # bottom row
output_coords = [(0.0, 1.0)]                                                # top
```

> A mismatch between the number of coordinates and the input/output length is the most common first
> error. Count the bias.

## Complete runnable example

This evolves a network to solve XOR using a per-node `sin` activation. It is self-contained and
solves on the first generation. (It prints `initializing` / `initializing finished` from the
substrate build, which is expected.)

```python
from emr_hyperneat import EMRHyperNEAT

# 1) Define your problem.
class XorProblem:
    input_shape  = (3,)        # 2 inputs + 1 bias
    output_shape = (1,)
    jitable = True
    use_bias = True
    fitness_threshold = 0.95
    def get_data(self):
        return [([0., 0., 1.], [0.]), ([0., 1., 1.], [1.]),
                ([1., 0., 1.], [1.]), ([1., 1., 1.], [0.])]

problem = XorProblem()

# 2) One substrate coordinate per input node and per output node.
n_in = problem.input_shape[0]
input_coords  = [(-1.0 + 2.0 * i / (n_in - 1), -1.0) for i in range(n_in)]
output_coords = [(0.0, 1.0)]

# 3) Config: enable a global 'sin' activation (oscillatory -> solves XOR fast).
config = {"algorithm_params": {"emrhyperneat": {
    "population_size": 150,
    "substrate": {"input_coords": input_coords, "output_coords": output_coords},
    "emr_hyperneat": {
        "initial_depth": 0, "max_depth": 2, "variance_threshold": 0.03,
        "dynamic_functions": {"mode": "global", "hidden_activation": "sin"},
    },
}}}

# 4) Evolve.
algo  = EMRHyperNEAT()
cfg   = algo.create_config(config)
state = algo.initialize(cfg, problem, seed=0)

best = 0.0
for gen in range(20):
    state, metrics = algo.run_generation(state, problem, verbose=False)
    best = max(best, float(metrics.best_fitness))
    if best >= problem.fitness_threshold:
        print(f"solved at gen {gen}: best_fitness={best:.6f}")
        break
```

Run it on CPU:

```bash
JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3 python my_experiment.py
```

Swap `"sin"` for `"tanh"` and you will watch a monotonic activation struggle on the same task. That
contrast is the subject of one of the papers.

### What to expect on the first run

- **The first generation compiles.** JAX JIT-compiles the substrate pipeline, so the first
  `run_generation` pauses a few seconds and the algorithm prints `initializing` / `initializing
  finished`. That is normal, not a hang; later generations are fast.
- **Runs are deterministic on a CPU.** The same `seed` and config produce identical results every
  time. On a GPU, a sum taken with a scatter-add accumulates in no fixed order, so a configuration
  that sums that way does not repeat bit for bit. Dynamic h→h aggregation (`hh_aggregation_mode:
  dynamic`) sums that way: on an RTX 2080 Ti, two evaluations of one generation put a genome's
  fitness up to 8.9e-4 apart, which can change which genomes are selected.
- **Fitness is in `[0, 1]`** (1.0 = a perfect solution). In a `run_generation` loop, "solved" is
  your own check, `best_fitness >= fitness_threshold` — the loop above decides when to stop. If you
  would rather hand that decision to the algorithm, `run_until_threshold(state, problem,
  target_fitness, max_generations)` owns the loop and stops itself.

## Switching on features

Each feature is a sub-dict under `emr_hyperneat`. The full option tables are in the
[configuration reference](configuration.md); the common starting points are:

```python
# Per-node activation chosen by the CPPN from a palette (here the 4 oscillatory functions):
"dynamic_functions": {"mode": "cppn_output", "palette": "oscillatory"},

# Recurrence (memory), hidden-to-hidden connections:
"recurrence": {"preset": "hidden_only"},

# Neuromodulation (needed for the multi-task API):
"neuromodulation": {"enabled": True, "mode": "true_neuromodulation", "num_nt_types": 4},
```

## Multi-task: one substrate, several tasks

One evolved substrate can serve several tasks, with a per-task neurotransmitter vector
`[DA, 5HT, NE, ACh]` switching its behavior. All tasks must share input/output shapes (it is one
substrate). Enable `true_neuromodulation` plus the `multitask` sub-dict, then drive the loop with
`run_generation_multitask`:

```python
import jax.numpy as jnp
from emr_hyperneat import EMRHyperNEAT

class GateProblem:                      # XOR and AND share shapes -> one substrate fits both
    input_shape, output_shape = (3,), (1,)
    jitable = True; use_bias = True; fitness_threshold = 0.95
    is_exhaustive_dataset = True
    def __init__(self, targets): self.targets = targets
    def get_data(self):
        ins = [[0.,0.,1.], [0.,1.,1.], [1.,0.,1.], [1.,1.,1.]]
        return [(i, [t]) for i, t in zip(ins, self.targets)]

problems   = [GateProblem([0.,1.,1.,0.]), GateProblem([0.,0.,0.,1.])]   # XOR, AND
nt_vectors = [jnp.array([0.95, 0.05, 0.95, 1.0]),                        # per-task [DA,5HT,NE,ACh]
              jnp.array([0.10, 0.90, 0.10, 1.0])]

n_in = problems[0].input_shape[0]
config = {"algorithm_params": {"emrhyperneat": {
    "population_size": 150,
    "substrate": {"input_coords": [(-1.0 + 2.0*i/(n_in-1), -1.0) for i in range(n_in)],
                  "output_coords": [(0.0, 1.0)]},
    "emr_hyperneat": {
        "max_depth": 2, "variance_threshold": 0.03,
        "neuromodulation": {"enabled": True, "mode": "true_neuromodulation", "num_nt_types": 4},
        "multitask": {"enabled": True, "num_tasks": 2, "task_names": ["xor", "and"],
                      "fitness_aggregation": "mean"},
    },
}}}

algo  = EMRHyperNEAT()
state = algo.initialize(algo.create_config(config), problems[0], seed=0)   # shapes from task 0
for gen in range(30):
    state, metrics = algo.run_generation_multitask(
        state, problems, neurotransmitters=nt_vectors, aggregation_method="mean")
    if metrics.best_fitness >= problems[0].fitness_threshold:
        break
```

Expect this to need more generations than single-task XOR: holding several tasks in one substrate
is the hard version of the problem (the ALIFE neuromodulation paper's multi-task barrier is exactly
this). The CPPN evolves once per generation; fitness is computed on every task with its NT vector
and combined by `aggregation_method` (`mean`, `min`, `weighted`, `product`, `softmin`, `harmonic` —
`min` demands a generalist, `mean` tolerates specialists). Omit `neurotransmitters` to use the
per-task vectors from the config (`multitask.nt_per_task`) or the named-task presets. The full key
list is in the [configuration reference](configuration.md#feature-3-neuromodulation-multi-task);
the runners under `papers/emr-neuromodulation/scripts/runners/` are complete worked examples.

## Reusing the test helpers

The test suite ships ready-made problems and a run loop you can copy:

- `emr_hyperneat/tests/conftest.py` defines `XORProblem`, `ParityProblem(n_bits)`,
  `MultiTaskProblem(task_name)`, and the `run_quick_evolution(...)` helper (the canonical loop).
- The paper runners under `papers/*/scripts/runners/` are the most complete examples. Each builds a config,
  defines or imports a problem, and evolves. They are the reference for real experiments.

Next: the [configuration reference](configuration.md) for every knob, or
[reproducing the paper experiments](reproducing-experiments.md).
