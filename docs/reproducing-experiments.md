# Reproducing the paper experiments

Each paper has a self-contained directory under `papers/` with its own runners, analysis scripts,
figures, and a README giving exact commands. This page explains the shared mechanics: how to get the
result data, and how the experiments map onto the algorithm.

## Choosing a release

Each paper names the release its code was published with: the four papers below name **0.1.0**
(tag `v0.1.0`). A later paper names the first release that contains its folder, and its runners
reproduce its recorded runs on that release, which its
[paper-validation tests](testing.md#the-paper-validation-tests) check. Release tags never move.
The code of a paper's runners and analysis scripts is not edited after its release (0.2.0 rewrote
only their comments and docstrings: apart from docstrings, their syntax trees match 0.1.0's), with
two exceptions: in 0.2.0 the scripts that ran the HMR engine became EMR versions, since 0.2.0 no
longer ships that engine, and the neuromodulation validator `probe_emr_config.py`, which failed at
import in 0.1.0, finds the shared runner. Each paper's README names them and how their published
runs reproduce on 0.1.0.

**With the script.** `scripts/replicate.py` builds a release in its own checkout,
`.replicate/v<release>/`: a git worktree of the release tag, with the TensorNEAT submodule and a
virtual environment installed as in [installation](installation.md) (about 2 GB, with network
access for pip and the submodule). Commands then run there; the current checkout and environment
are untouched.

```bash
python scripts/replicate.py list                       # each paper, its release, set up or not
python scripts/replicate.py run --release 0.1.0 -- python scripts/fetch_results.py
python scripts/replicate.py run emr-dynamic-functions -- python scripts/runners/E1_parity_scaling.py
python scripts/replicate.py setup emr-neuromodulation  # prints how to work there
```

With a paper, a command runs in that paper's folder of the release checkout, where the paper's
README gives its commands; with only `--release`, at the checkout's root. `--release <version>`
runs a paper on another release, and `--release current` on this checkout, with the Python that
runs the script; both print a note that names the paper's own release. The environment needs Python
3.10 to 3.12 (the lock pins NumPy 1.26.4); `--python` picks the interpreter. Remove a release
checkout with `git worktree remove --force .replicate/v0.1.0`.

**With git only.** Clone the release tag and install it as usual:

```bash
git clone --recursive --branch v0.1.0 https://github.com/RomainClaret/emr-hyperneat.git
```

In an existing clone, `git worktree add ../emr-hyperneat-0.1.0 v0.1.0` followed by
`git -C ../emr-hyperneat-0.1.0 submodule update --init` gives a second checkout, and
`git switch --detach v0.1.0 && git submodule update` switches this one (an editable install
follows the checkout). Use one environment per release.

**What 0.2.0 changes for these papers.** `replicate.py` runs each paper on 0.1.0 unless told
otherwise. On 0.2.0, on a CPU:

- The per-node paper's runners that use the `EMRHyperNEAT` class (E1, E2, E7, E8, E9;
  [below](#which-experiments-use-which-engine)) give the same fitness and evaluations as on 0.1.0,
  bit for bit, on arm64 and x86 processors. Their per-generation metrics now also report the
  population's species and topology, which 0.1.0 left empty.
- The neuromodulation paper's multi-head runner, EMR by default in 0.1.0 and EMR only in 0.2.0,
  departs from 0.1.0:
  it passes `neat_species` (compatibility 2.5, stagnation 40), which 0.1.0 ignored and 0.2.0
  applies. A run for XOR and AND at population 50, seed 42, matches 0.1.0 through generation 4
  and differs from generation 5; with `neat_species` set to 0.1.0's defaults (3.0 and 20) it
  matches 0.1.0 bit for bit. The paper's numbers come from the HMR engine, which the runner of
  release 0.1.0 can select through its `backend` parameter and which 0.2.0 removes.
- The runners whose published runs used the HMR engine are EMR versions, so their results differ
  from the published runs: each paper's README gives the comparison, and the published runs
  reproduce on 0.1.0.
- The pinned stack is unchanged, and so are the analysis scripts, but for the per-node paper's
  `_validate_emr_port.py`, which compares EMR with HMR's trajectories recorded on 0.1.0, and the
  neuromodulation validator's import of the shared runner. 0.2.0 does not ship the frozen HMR
  module. A run in which a genome's fitness is NaN can differ, since 0.2.0 ranks that genome last
  (in 0.1.0 the NaN reached selection).

The paper-validation tests check each paper's recorded values on every release.

## Step 1: get the result data

The repository ships **code only**. The experiment result data (the JSON the analysis and figure
scripts read) is a separate **data release** on Zenodo (~9 MB compressed, ~180 MB unpacked):

[![Data release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383893.svg)](https://doi.org/10.5281/zenodo.21383893)

Fetch it once:

```bash
python scripts/fetch_results.py            # download from Zenodo, verify md5, unpack into papers/
```

> **Offline copy:** if you already have the archive locally, skip the download:
> `python scripts/fetch_results.py --archive <path-to-archive.tar.gz>`.

`fetch_results.py` downloads one archive, checks the md5 Zenodo publishes for it, and unpacks only the `papers/`-prefixed
members into each paper's `results/` and `data/` folders. Useful flags:

| Flag | Effect |
|---|---|
| (none) | Download + verify + unpack. |
| `--archive <path>` | Use a local copy of the archive (offline). |
| `--check` | Verify an already-downloaded archive without unpacking. |
| `--keep` | Keep the temporary download. |

The result/data folders are gitignored, which is why a fresh clone has empty `results/`. The archive
is on Zenodo, latest version via the concept DOI ([10.5281/zenodo.21383893](https://doi.org/10.5281/zenodo.21383893)); pass `--archive <path>` to use
a local copy offline.

> **One paper is different.** `papers/emr-hyperneat` ships **raw measurement data** under `data/` in
> the same data release (fetched, not committed, `data/` is gitignored too), and its tables
> regenerate from that data with **no GPU and no evolution**. The other three papers read from
> `results/`.

## Step 2: reproduce one paper

```bash
pip install -e ".[analysis]"        # scipy + matplotlib for the analysis/figure scripts
cd papers/<paper>
cat README.md                       # the exact, per-paper commands
python <analysis_or_figure_script>.py
```

## The four papers

| Directory | Venue | Release | PDF | Adds | How you reproduce it |
|---|---|---|---|---|---|
| [`emr-hyperneat`](../papers/emr-hyperneat/README.md) | GECCO 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-hyperneat) | [PDF](https://claret.tech/pdf/claret2026emr) | the base EMR algorithm + GPU speedups | Regenerate the speedup/scaling tables from the fetched `data/` with `scripts/analysis/compute_speedup.py`, `scripts/analysis/extract_*_iqr.py`, `scripts/analysis/analyze_benchmark_data.py`; the two poster figures are standalone TikZ (`scripts/figures/*.tex`, compiled with `pdflatex`, see the paper's README). No GPU needed. |
| [`emr-dynamic-functions`](../papers/emr-dynamic-functions/README.md) | ALIFE 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions) | [PDF](https://claret.tech/pdf/claret2026activations) | per-node activation evolution | Runners `scripts/runners/E1..E9_*.py` write to `results/`; analyze with `scripts/analysis/analysis_paper_statistics.py`. See its `REPRODUCIBILITY.md`. |
| [`emr-dynamic-functions-bio-inspired`](../papers/emr-dynamic-functions-bio-inspired/README.md) | PPSN 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions-bio-inspired) | [PDF](https://claret.tech/pdf/claret2026bio) | bio-inspired palette evolution | The strategies live in `scripts/palette_strategies/`; runners `scripts/runners/benchmark_bio_*.py`; verify with `scripts/analysis/verify_paper_numbers.py`. See `BENCHMARK_RESULTS.md`. |
| [`emr-neuromodulation`](../papers/emr-neuromodulation/README.md) | ALIFE 2026 | [0.1.0](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-neuromodulation) | [PDF](https://claret.tech/pdf/claret2026neuromodulation) | neuromodulation / multi-task | Most `scripts/runners/benchmark_*.py` are self-contained; the indirect-encoding headline uses `scripts/runners/multihead_palette_neuromodulation.py`. See `EXPERIMENT_RESULTS.md`. |

## Which experiments use which engine

Not every published number came from the current `EMRHyperNEAT` class. Some came from the **HMR
engine**, the **frozen HMR module** that release 0.1.0 ships (`emr_hyperneat/_hmr_frozen/`
there; 0.2.0 removed it), which is bit-for-bit equivalent to EMR on the shared paths when EMR is
pinned to `variance_rule: legacy` (the package default is `canonical`) and given no `neat_*`
settings, and reproduces the exact numbers in the published tables on that release:

- **EMR (current class), published on EMR:** the `emr-dynamic-functions` E1/E2/E7/E8/E9 runners.
- **EMR from 0.2.0, published on the HMR engine:** the `emr-neuromodulation` indirect-encoding
  runner; the `emr-dynamic-functions` E3/E4/E5 runners; the `emr-dynamic-functions-bio-inspired`
  bio-palette runners; and the `emr-hyperneat` optional from-scratch runners. Each runs its
  experiment as it ran, on EMR's defaults apart from HMR's level-0 variance rule (and, for the
  runners the frozen module ran, its random key split), which it pins, and writes apart from the
  published runs; the published runs reproduce on 0.1.0
  (`python scripts/replicate.py run <paper> --release 0.1.0 -- <command>`).
- **Checked against HMR's recorded values:** the `emr-dynamic-functions` validator
  `scripts/analysis/_validate_emr_port.py` compares EMR with HMR's trajectories of its 18 cells,
  recorded on 0.1.0 (`scripts/analysis/hmr_trajectories.json`).
- **Neither (plain JAX):** the `emr-dynamic-functions` E6 gradient-descent baseline.

The per-paper `REPRODUCIBILITY.md` / `BENCHMARK_RESULTS.md` / `EXPERIMENT_RESULTS.md` files are the
authoritative map of runner -> engine -> finding.

## Using a GPU

You do **not** need a GPU to reproduce any published table or figure. They all regenerate from the
fetched data on CPU. A GPU only matters if you want to *re-measure* the substrate-discovery speedups
yourself (the GECCO paper's headline). For that:

- Install a CUDA build of JAX and set `JAX_PLATFORMS=gpu` (see
  [installation](installation.md#3-pick-a-backend)).
- Select a multi-GPU strategy in the config under `emr_hyperneat` (`multi_gpu_strategy`), which needs
  2+ visible devices; the worker plumbing lives in `emr_hyperneat/multi_gpu_worker.py`.
- The `papers/emr-hyperneat` from-scratch runners (`scripts/runners/run_gecco2026_experiments.py`,
  `scripts/runners/run_statistical_validation.py`) are the worked examples. They run EMR-HyperNEAT
  from 0.2.0 (their published files came from the HMR engine) and require a GPU.

Multi-GPU needs 2+ visible devices and engages **only through `run_until_threshold`** (a
`run_generation` loop stays single-device). `full_pipeline_parallel` runs on one device
instead when the problem sets `is_exhaustive_dataset` or has fewer than 16 samples per
device; the other strategies ignore the flag. What each strategy splits and replicates:
[architecture](architecture.md#what-the-multi-gpu-strategies-parallelize). No published table needs
any of this.

## Don't have the data? Validate the findings anyway

The [paper-validation tests](testing.md#the-paper-validation-tests) re-run the *simple* version of
each experiment fresh (no downloaded data needed) and assert the headline finding:

```bash
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_paper_validation.py -m paper -q
```

That is the fastest way to check that the algorithm still gives each paper's headline finding.
