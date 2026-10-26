# EMR-HyperNEAT documentation

This is the documentation for using and extending **EMR-HyperNEAT** (Eager Multi-Resolution
HyperNEAT), a GPU-accelerated reformulation of ES-HyperNEAT for adaptive-substrate neuroevolution.

If you are new here, read the pages in this order:

| # | Page | What you get |
|---|------|--------------|
| 1 | [Installation](installation.md) | Clone with the TensorNEAT submodule, install the pinned JAX stack, run on CPU or GPU. |
| 2 | [Architecture](architecture.md) | A component diagram and the module map, what layer EMR-HyperNEAT adds on top of TensorNEAT, and what each module does. |
| 3 | [Writing experiments](writing-experiments.md) | The public API, the `Problem` interface, and a complete runnable example that evolves a network to solve XOR. |
| 4 | [Configuration reference](configuration.md) | Every config knob: the substrate, and the four feature dimensions (dynamic activations, aggregation, neuromodulation, recurrence). |
| 5 | [Testing](testing.md) | Run the test suite, the markers, and validate the GitHub Action locally with `act`. |
| 6 | [Reproducing the paper experiments](reproducing-experiments.md) | Fetch the data release and regenerate each paper's tables and figures. |

## The shortest path

```bash
git clone --recursive https://github.com/RomainClaret/emr-hyperneat.git
cd emr-hyperneat
pip install -e . -c requirements-lock.txt
pip install -e third_party/tensorneat -c requirements-lock.txt
export JAX_PLATFORMS=cpu          # CPU runs the library, the tests, and every published table
```

Then evolve a network in about 30 lines (see [Writing experiments](writing-experiments.md)).

## Status

This is research code, not production software. It is tested for reproducing the published findings
and for the experiments documented here; anything else is untested. See the
[root README](../README.md#status) for the full caveat, and
[keys that parse but do not act](configuration.md#keys-that-parse-but-do-not-act) before relying on
an undocumented knob.
