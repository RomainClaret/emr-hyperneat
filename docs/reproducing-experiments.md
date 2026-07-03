# Reproducing the paper experiments

Each paper has a self-contained directory under `papers/` with its own runners, analysis scripts,
figures, and a README giving exact commands. This page explains the shared mechanics: how to get the
result data, and how the experiments map onto the algorithm.

## Step 1: get the result data

The repository ships **code only**. The experiment result data (the JSON the analysis and figure
scripts read) is a separate **data release** on Zenodo (~9 MB compressed, ~180 MB unpacked):

[![Data release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383893.svg)](https://doi.org/10.5281/zenodo.21383893)

Fetch it once:

```bash
python scripts/fetch_results.py            # download from Zenodo, verify sha256, unpack into papers/
```

> **Offline copy:** if you already have the archive locally, skip the download:
> `python scripts/fetch_results.py --archive <path-to-archive.tar.gz>`.

`fetch_results.py` downloads one archive, checks its sha256, and unpacks only the `papers/`-prefixed
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

| Directory | Venue | Adds | How you reproduce it |
|---|---|---|---|
| [`emr-hyperneat`](../papers/emr-hyperneat/README.md) | GECCO 2026 | the base EMR algorithm + GPU speedups | Regenerate the speedup/scaling tables from the fetched `data/` with `scripts/analysis/compute_speedup.py`, `scripts/analysis/extract_*_iqr.py`, `scripts/analysis/analyze_benchmark_data.py`; the two poster figures are standalone TikZ (`scripts/figures/*.tex`, compiled with `pdflatex`, see the paper's README). No GPU needed. |
| [`emr-dynamic-functions`](../papers/emr-dynamic-functions/README.md) | ALIFE 2026 | per-node activation evolution | Runners `scripts/runners/E1..E9_*.py` write to `results/`; analyze with `scripts/analysis/analysis_paper_statistics.py`. See its `REPRODUCIBILITY.md`. |
| [`emr-dynamic-functions-bio-inspired`](../papers/emr-dynamic-functions-bio-inspired/README.md) | PPSN 2026 | bio-inspired palette evolution | The strategies live in `scripts/palette_strategies/`; runners `scripts/runners/benchmark_bio_*.py`; verify with `scripts/analysis/verify_paper_numbers.py`. See `BENCHMARK_RESULTS.md`. |
| [`emr-neuromodulation`](../papers/emr-neuromodulation/README.md) | ALIFE 2026 | neuromodulation / multi-task | Most `scripts/runners/benchmark_*.py` are self-contained; the indirect-encoding headline uses `scripts/runners/multihead_palette_neuromodulation.py`. See `EXPERIMENT_RESULTS.md`. |

## Which experiments use which engine

Not every paper runner uses the current `EMRHyperNEAT` class. Some reproduce pre-migration results on
the **frozen HMR module** (`emr_hyperneat/_hmr_frozen/`), which is bit-for-bit equivalent on the
shared paths (see [architecture](architecture.md#module-map)):

- **EMR (current class):** the `emr-dynamic-functions` E1/E2/E7/E8/E9 runners; the
  `emr-neuromodulation` indirect-encoding runner (`backend='emr'`, the default).
- **Frozen HMR:** the `emr-dynamic-functions` E3/E4/E5 runners; the `emr-dynamic-functions-bio-inspired`
  bio-palette runners; the `emr-hyperneat` optional from-scratch runners; and `backend='hmr'` for the
  exact published neuromodulation baseline.
- **Neither (plain JAX):** the `emr-dynamic-functions` E6 gradient-descent baseline.

The per-paper `REPRODUCIBILITY.md` / `BENCHMARK_RESULTS.md` / `EXPERIMENT_RESULTS.md` files are the
authoritative map of runner → engine → finding.

## Using a GPU

You do **not** need a GPU to reproduce any published table or figure. They all regenerate from the
fetched data on CPU. A GPU only matters if you want to *re-measure* the substrate-discovery speedups
yourself (the GECCO paper's headline). For that:

- Install a CUDA build of JAX and set `JAX_PLATFORMS=gpu` (see
  [installation](installation.md#3-pick-a-backend)).
- Select a multi-GPU strategy in the config under `emr_hyperneat` (`multi_gpu_strategy`), which needs
  2+ visible devices; the worker plumbing lives in `emr_hyperneat/multi_gpu_worker.py`.
- The `papers/emr-hyperneat` from-scratch runners (`scripts/runners/run_gecco2026_experiments.py`,
  `scripts/runners/run_statistical_validation.py`) are the worked examples. They run on the frozen HMR
  module and require a GPU.

The multi-GPU path is not exercised by this CPU-only documentation; treat it as advanced.

## Don't have the data? Validate the findings anyway

The [paper-validation tests](testing.md#the-paper-validation-tests) re-run the *simple* version of
each experiment fresh (no downloaded data needed) and assert the headline finding:

```bash
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_paper_validation.py -m paper -q
```

That is the fastest way to confirm the algorithm still reproduces what the papers claim.
