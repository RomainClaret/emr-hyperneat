# EMR-HyperNEAT

**Eager Multi-Resolution HyperNEAT**, a GPU-accelerated reformulation of
ES-HyperNEAT for adaptive-substrate neuroevolution.

ES-HyperNEAT discovers where to place neurons by recursively subdividing space with
a **sequential quadtree** driven by CPPN-output variance. That quadtree is inherently
serial and CPU-bound, which caps the method at small substrates. EMR-HyperNEAT
replaces it with **eager tensor evaluation on a pre-computed hierarchical grid** plus
**post-hoc variance masking** by the same criterion: substrate discovery becomes batch matrix
arithmetic that `vmap`s across an entire population on the GPU. On XOR at population 1000,
EMR-HyperNEAT on an RTX 2080 Ti is slower than a feed-forward port of ES-HyperNEAT on an M4 Max
CPU at depths 1 to 3 and more than an order of magnitude faster per generation at depth 7, with
the h->h connection cache on or off.

<p align="center">
  <img src="docs/figures/runtime_iqr_comparison.png" width="720" alt="Runtime scaling on XOR: ES-HyperNEAT vs EMR-HyperNEAT across substrate depths, on CPU and GPU">
</p>
<p align="center"><em>Wall-clock time of 30 generations on XOR as substrate depth grows. ES-HyperNEAT's sequential quadtree (orange, CPU) becomes intractable past depth 7, while EMR-HyperNEAT's vectorized discovery keeps scaling: on CPU (green) through depth 9 and on GPU (blue) out to depth 13. A run's total is its start-up time plus 30 times its steady time per generation. Bands show the interquartile range of each depth's totals over population sizes and seeds; the GPU at depths 1 to 7 and ES-HyperNEAT at depths 1 to 6 have one value per population size. The runs at depths 8 and 9 on the CPU and 8 to 13 on the GPU lasted 10 generations (6 for the population-300 GPU seed that solved XOR), so their totals are projections. The CPU candlesticks at depth 10 project every population from three runs at population 50. The GPU band at depth 7 includes an estimated cell (population 1000) and an interpolated one (population 750); the band at depth 13 includes a seed projected from depth 12. EMR ran h->h connections with the connection cache on, except on the GPU at depths 8 to 13, which ran feed-forward. Adapted from the GECCO 2026 paper's runtime figure, whose CPU band ends at depth 7.</em></p>

This repository packages the EMR-HyperNEAT algorithm as a standalone, installable
library together with the code to reproduce the experiments from its papers.

**Read the papers (PDF):** [EMR-HyperNEAT (GECCO 2026)](https://claret.tech/pdf/claret2026emr) ·
[Per-node activations (ALIFE 2026)](https://claret.tech/pdf/claret2026activations) ·
[Bio-inspired palettes (PPSN 2026)](https://claret.tech/pdf/claret2026bio) ·
[Neuromodulation (ALIFE 2026)](https://claret.tech/pdf/claret2026neuromodulation)

## What it does

On top of the substrate engine, the package makes the substrate's *computation* evolvable along
four dimensions: **per-node activation functions** (an 18-function palette instead of one fixed
nonlinearity), **per-node aggregation** (global, CPPN-assigned or weight-interpreted),
**neuromodulation** (`[DA, 5HT, NE, ACh]` vectors
let one substrate serve several tasks), and **hidden-to-hidden recurrence** with an optional
connection cache (off by default since 0.2.0; see [docs/configuration.md](docs/configuration.md)).
Deep substrates run through
memory-bounded execution modes and optional multi-GPU strategies. The mechanism, end to end:
[docs/architecture.md](docs/architecture.md#how-a-generation-is-computed).

## Documentation

Full guides live in [`docs/`](docs/README.md):

| Guide | For |
|---|---|
| [Installation](docs/installation.md) | Cloning with the submodule, the pinned JAX stack, CPU/GPU. |
| [Architecture](docs/architecture.md) | The component diagram, what EMR adds on top of TensorNEAT, the module map. |
| [Writing experiments](docs/writing-experiments.md) | The public API, the `Problem` interface, a complete runnable example. |
| [Configuration reference](docs/configuration.md) | Every knob: the substrate and the four feature dimensions. |
| [Testing](docs/testing.md) | The test suite, markers, and validating CI locally with `act`. |
| [Reproducing experiments](docs/reproducing-experiments.md) | Fetching the data release and regenerating each paper's results. |
| [Changelog](CHANGELOG.md) | What this release contains. |

## Status

> **Research code, not production software.** This repository is the code accompanying the
> EMR-HyperNEAT papers. It is tested for reproducing their published results and for showing
> what the EMR-HyperNEAT concept enables; other uses are untested.

## How it fits together

EMR-HyperNEAT keeps ES-HyperNEAT's CPPN-based encoding but swaps its sequential quadtree for an
eager, vectorized substrate-discovery pipeline:

<p align="center">
  <img src="docs/figures/algorithm_comparison.png" width="760" alt="ES-HyperNEAT lazy sequential discovery versus EMR-HyperNEAT eager vectorized discovery">
</p>
<p align="center"><em>ES-HyperNEAT discovers nodes lazily, one genome and one position at a time (a quadratic number of sequential CPPN queries). EMR-HyperNEAT pre-computes the candidate grid once, queries every position for every genome in one vectorized pass, then filters by variance, which is what makes substrate discovery GPU-parallel. The GECCO 2026 paper's Figure 1 draws the same contrast on the grid itself.</em></p>

The library wraps that pipeline behind one public class, layered on top of TensorNEAT:

```mermaid
flowchart TB
    U["<b>Your code</b><br/>Problem.get_data() + config dict"] --> API["<b>EMRHyperNEAT</b><br/>public API"]
    API --> EMR["<b>The EMR layer</b> (what this project adds)<br/>eager multi-resolution substrate discovery + variance masking ·<br/>per-node dynamic activations · neuromodulation · recurrence · multi-GPU"]
    EMR --> SHIM["<b>_compat</b> shim, runs standalone, no external framework"]
    SHIM --> TN["<b>third_party/tensorneat</b>, NEAT / CPPN engine (black box)"]
    API -. "metrics (best_fitness, …)" .-> U
```

See [Architecture](docs/architecture.md) for the full diagram, the module map, and the
TensorNEAT integration seam.

## Installation

```bash
git clone --recursive https://github.com/RomainClaret/emr-hyperneat.git
cd emr-hyperneat
pip install -e . -c requirements-lock.txt                      # the emr_hyperneat package
pip install -e third_party/tensorneat -c requirements-lock.txt # the pinned TensorNEAT fork
export JAX_PLATFORMS=cpu                                        # CPU runs the library, the tests, and every published table
```

Requires Python ≥ 3.10. The `-c requirements-lock.txt` constraint pins the exact stack the published
results were produced on (JAX 0.6.1); omit it to run on any JAX in the supported range
(0.5-0.6.x, see `pyproject.toml`); the bit-exact golden tests auto-skip off 0.6.x. Full instructions, GPU setup, and troubleshooting:
[docs/installation.md](docs/installation.md).

## Quick start

Evolving takes one class and four calls. A problem is any plain object exposing
`get_data()` and its shapes:

```python
from emr_hyperneat import EMRHyperNEAT

algo  = EMRHyperNEAT()
cfg   = algo.create_config(config)               # substrate coordinates + feature switches
state = algo.initialize(cfg, problem, seed=0)    # builds the substrate, JIT-compiles
state, metrics = algo.run_generation(state, problem)
```

A complete runnable XOR example (a global `sin` activation solves it in the first generations),
the `Problem` interface, and multi-task usage: [docs/writing-experiments.md](docs/writing-experiments.md).
To inspect what evolved, `algo.population_substrates(state)` returns each genome's feed-forward
substrate exactly as the generation loop builds it
([the substrate on its own](docs/architecture.md#the-substrate-on-its-own)).

## Testing

```bash
pip install -e ".[dev]"
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -m "not slow" -q   # the non-slow suite (what the nightly runs)
```

The suite covers every published feature plus isolation, determinism, and EMR-vs-HMR
equivalence guards (against values recorded from HMR), and includes paper-validation tests that
re-run each paper's simplest experiment. Full commands, markers, the Linux `vm.max_map_count` note,
and local-CI instructions:
[docs/testing.md](docs/testing.md).

## The papers

Each subdirectory of [`papers/`](papers/) is self-contained (its own runners, analysis, figures, and
a README with exact reproduction commands). Each paper adds a capability to the same algorithm core.

| Paper | Venue | Release | Adds | PDF | Publication |
|-------|-------|---------|------|-----|-------------|
| [`emr-hyperneat`](papers/emr-hyperneat/README.md) | GECCO 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-hyperneat) | the base EMR algorithm + GPU speedups | [PDF](https://claret.tech/pdf/claret2026emr) | [10.1145/3795101.3805361](https://doi.org/10.1145/3795101.3805361) |
| [`emr-dynamic-functions`](papers/emr-dynamic-functions/README.md) | ALIFE 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions) | per-node activation function evolution | [PDF](https://claret.tech/pdf/claret2026activations) | [10.1162/ISAL.a.1042](https://doi.org/10.1162/ISAL.a.1042) |
| [`emr-dynamic-functions-bio-inspired`](papers/emr-dynamic-functions-bio-inspired/README.md) | PPSN 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions-bio-inspired) | bio-inspired palette-evolution strategies | [PDF](https://claret.tech/pdf/claret2026bio) | [10.1007/978-3-032-36217-9_23](https://doi.org/10.1007/978-3-032-36217-9_23) |
| [`emr-neuromodulation`](papers/emr-neuromodulation/README.md) | ALIFE 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-neuromodulation) | neuromodulation for multi-task learning | [PDF](https://claret.tech/pdf/claret2026neuromodulation) | [10.1162/ISAL.a.968](https://doi.org/10.1162/ISAL.a.968) |

**Each paper names the release its code was published with** (a later paper names the first
release that contains its folder). `scripts/replicate.py` runs a paper's commands on that release,
in its own checkout and environment under `.replicate/`:

```bash
python scripts/replicate.py run emr-dynamic-functions -- python scripts/runners/E1_parity_scaling.py
```

`--release <version>` picks another release, and `--release current` this checkout. On a CPU,
0.2.0 gives the per-node paper's runners the same results as 0.1.0; the neuromodulation paper's
runner departs from 0.1.0's default EMR backend, since 0.2.0 applies its species settings (the
paper's numbers come from the HMR engine, which release 0.1.0 ships and 0.2.0 does not).
The tables and figures regenerate from the data release with any release. Details, and the
plain-git route:
[docs/reproducing-experiments.md](docs/reproducing-experiments.md#choosing-a-release).

The repository ships **code only**; result data is a separate Zenodo release fetched with
`python scripts/fetch_results.py`. How to reproduce each paper, and which published numbers came
from the current EMR class or the HMR engine: [docs/reproducing-experiments.md](docs/reproducing-experiments.md).

Data archive (all papers' result JSON), on Zenodo: [![Data release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383893.svg)](https://doi.org/10.5281/zenodo.21383893)

Some published tables came from the **HMR engine**, a frozen module of release 0.1.0, which
reproduces them exactly; 0.2.0 does not ship it. In 0.2.0 their runners are EMR versions,
and the tests and the per-node paper's validator compare EMR with values recorded from HMR. HMR
is bit-for-bit equivalent to EMR on the shared paths when EMR is pinned to `variance_rule: legacy`
(the package default is `canonical`) and is given no `neat_*` settings.

### Scope

- **Paper features:** the EMR substrate machinery (GECCO), per-node activations and the
  18-function palette (ALIFE), palette meta-learning with per-node aggregation (PPSN),
  and neuromodulation with multi-task evaluation (ALIFE).
- **Infrastructure:** the execution modes, memory budgets, compaction, and multi-GPU strategies
  make the deep-substrate results runnable; they change cost, and results at most by float32
  rounding (as compaction does on x86 CPUs and GPUs), except `truncate_variance_lookahead`,
  off by default, which changes the substrate.
- **Extras in no paper:** `geometry_seeding` (tested, documented, off by default) and the
  programmatic multi-task options `two_module_mode` / `generalist_bonus_type`. No published result
  depends on them. See [configuration](docs/configuration.md) for each one's exact status.

## Repository layout

```
emr_hyperneat/        installable algorithm package (public API: EMRHyperNEAT, substrate)
  engine/             the engine's code by concern (internal; emrhyperneat.py re-exports it)
  _compat/            vendored compatibility shim (internal; lets EMR run standalone)
  tests/              test suite incl. paper-validation
docs/                 documentation (start at docs/README.md)
third_party/tensorneat  pinned TensorNEAT fork (git submodule)
papers/               one self-contained directory per paper
scripts/              utilities (e.g. fetch_results.py)
```

## Community & Support

Start a thread in [GitHub Discussions](https://github.com/RomainClaret/emr-hyperneat/discussions)
to ask for help, discuss the algorithm and its results, or share what you built or found.
For concrete bugs or problems, open a [GitHub issue](https://github.com/RomainClaret/emr-hyperneat/issues).

## License

Copyright (C) 2026 Romain Claret

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

## Acknowledgements

EMR-HyperNEAT builds directly on prior work:

- **ES-HyperNEAT** ([Risi and Stanley, 2012](https://doi.org/10.1162/artl_a_00071)), the evolvable-substrate method that EMR-HyperNEAT reformulates.
- **TensorNEAT** ([Wang et al., 2025](https://doi.org/10.1145/3730406)), the GPU-accelerated NEAT/CPPN library this project builds on and vendors as a submodule.
- **PUREPLES** ([Westh et al., 2017](https://github.com/ukuleleplayer/pureples)), whose ES-HyperNEAT implementation inspired parts of this one.

## Citation

The first entry below is **EMR-HyperNEAT itself**, the paper that introduces the
algorithm. If you use this repository, please cite it. The entries after it are the
papers that **extend EMR-HyperNEAT** with additional capabilities (per-node activation
functions, bio-inspired palette meta-learning and neuromodulation); if you use one of those
features, please also cite the corresponding paper.

**The EMR-HyperNEAT algorithm** ([PDF](https://claret.tech/pdf/claret2026emr))**:**

```bibtex
@inproceedings{claret2026emr,
  title={Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO Companion '26)},
  year={2026},
  address={San Jose, Costa Rica},
  publisher={ACM},
  doi={10.1145/3795101.3805361},
  url={https://claret.tech/pdf/claret2026emr}
}
```

**Papers extending EMR-HyperNEAT** ([activations](https://claret.tech/pdf/claret2026activations) ·
[bio-inspired palettes](https://claret.tech/pdf/claret2026bio) ·
[neuromodulation](https://claret.tech/pdf/claret2026neuromodulation))**:**

```bibtex
@inproceedings{claret2026activations,
  title={Per-Node Activation Function Evolution in Indirectly Encoded Substrates: Solvability, Limits, and Emergent Diversity},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={ALIFE 2026: Proceedings of the 2026 Artificial Life Conference},
  volume={38},
  pages={80},
  year={2026},
  publisher={MIT Press},
  doi={10.1162/ISAL.a.1042},
  url={https://claret.tech/pdf/claret2026activations}
}

@inproceedings{claret2026bio,
  title={Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility Shapes Activation Function Discovery},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={Parallel Problem Solving from Nature -- PPSN XIX},
  year={2026},
  organization={Springer},
  doi={10.1007/978-3-032-36217-9_23},
  url={https://claret.tech/pdf/claret2026bio}
}

@inproceedings{claret2026neuromodulation,
  title={Multi-Behavioral Evolved Substrates Through Neuromodulation and Activation Selection},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={ALIFE 2026: Proceedings of the 2026 Artificial Life Conference},
  volume={38},
  pages={78},
  year={2026},
  publisher={MIT Press},
  doi={10.1162/ISAL.a.968},
  url={https://claret.tech/pdf/claret2026neuromodulation}
}
```
