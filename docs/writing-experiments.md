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
| `run_until_threshold` | `run_until_threshold(state, problem, target_fitness, max_generations, collect_history=False)` | a dict with `generations`, `best_fitness`, `state`, and `history` (if requested) |
| `run_generation_multitask` | `run_generation_multitask(state, problems, neurotransmitters=None, aggregation_method='mean')` | `(new_state, metrics)`. Requires neuromodulation in `true_neuromodulation` mode |

### The metrics object

`run_generation` returns an `AlgorithmMetrics` with at least: `generation`, `best_fitness`,
`mean_fitness`, `min_fitness`, `max_fitness`, `std_fitness`, `evaluations`, `time_elapsed`, and a
`custom_metrics` dict. For EMR the useful `custom_metrics` keys are `total_positions` (the candidate
grid size), `avg_hidden_nodes` / `max_hidden_nodes` (how many survived variance masking), and
`position_utilization` (the sparse fraction).

## The Problem interface

A problem is duck-typed. Give it these attributes and one method:

```python
class MyProblem:
    input_shape  = (3,)     # (number of inputs,) INCLUDING the bias column if you use one
    output_shape = (1,)     # (number of outputs,)
    jitable = True          # your get_data() returns plain numbers, safe to JIT
    use_bias = True         # whether your inputs carry an explicit bias column
    fitness_threshold = 0.95   # what you consider "solved" (your loop decides when to stop)

    def get_data(self):
        # return a list of (input, target) pairs
        return [([0., 0., 1.], [0.]), ([0., 1., 1.], [1.]),
                ([1., 0., 1.], [1.]), ([1., 1., 1.], [0.])]
```

- `get_data()` returns a **list of `(input, target)` pairs**. Each `input` has length
  `input_shape[0]`, each `target` has length `output_shape[0]`. The pairs are stacked into arrays of
  shape `(num_cases, num_inputs)` and `(num_cases, num_outputs)`.
- If you use a bias, include an explicit `1.0` column in every input and count it in `input_shape`.

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
- **Runs are deterministic.** The same `seed` and config produce identical results every time.
- **Fitness is in `[0, 1]`** (1.0 = a perfect solution). "Solved" is just your own check,
  `best_fitness >= fitness_threshold`; the algorithm has no separate notion of "done".

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

For multi-task evolution (one substrate, several tasks) you pass a list of problems to
`run_generation_multitask` with neuromodulation enabled. The neuromodulation paper's runners under
`papers/emr-neuromodulation/scripts/runners/` are worked examples of that path.

## Reusing the test helpers

The test suite ships ready-made problems and a run loop you can copy:

- `emr_hyperneat/tests/conftest.py` defines `XORProblem`, `ParityProblem(n_bits)`,
  `MultiTaskProblem(task_name)`, and the `run_quick_evolution(...)` helper (the canonical loop).
- The paper runners under `papers/*/scripts/runners/` are the most complete examples. Each builds a config,
  defines or imports a problem, and evolves. They are the reference for real experiments.

Next: the [configuration reference](configuration.md) for every knob, or
[reproducing the paper experiments](reproducing-experiments.md).
