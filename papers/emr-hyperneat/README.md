# EMR-HyperNEAT (the algorithm paper)

**Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates**
Romain Claret, Michael O'Neill, Paul Cotofrei, Kilian Stoffel, GECCO 2026.

> Paper: **[PDF](https://claret.tech/pdf/claret2026emr)** · DOI [10.1145/3795101.3805361](https://doi.org/10.1145/3795101.3805361) (GECCO Companion '26). BibTeX in [Citation](#citation) below.
> The PDF is hosted at claret.tech, not committed to this repository.
> Code release: **emr-hyperneat 0.1.0** (tag [`v0.1.0`](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-hyperneat)); run a command below on it with `python scripts/replicate.py run emr-hyperneat -- <command>` from the repository root ([choosing a release](../../docs/reproducing-experiments.md#choosing-a-release)).

This is the paper that introduces **EMR-HyperNEAT** itself: replacing ES-HyperNEAT's
sequential, CPU-bound quadtree with eager tensor evaluation on a pre-computed
hierarchical grid plus variance masking, so substrate discovery `vmap`s across a whole
population on the GPU. It reports the resulting speedups over ES-HyperNEAT (a feed-forward
Python port of PUREPLES's algorithm) across substrate depths and population sizes, the
depth/recurrence-preset studies, and the JIT-vs-steady-state timing breakdown.

## What this paper adds to the algorithm

The base `emr_hyperneat.EMRHyperNEAT`: hierarchical-grid substrate discovery,
variance-based masking, batched CPPN queries, and the two-level (JAX compilation + h->h
connection) caching. Everything the later papers build on.

## Reproducing the paper's tables and figures

The paper's quantitative content is regenerated **from the measurement data**:
no GPU and no evolution run required. Each runtime number is a 30-generation
`total_runtime_s` computed from per-run construction and steady per-generation times;
Timing provenance, below, says which of those times are measured and which runs are
projected from shorter ones. Fetch the data release first (see root
`scripts/fetch_results.py`), which populates `data/`. Then:

```bash
# Runtime IQR bands + speedups, from the per-run timing data (scripts/analysis/):
python scripts/analysis/extract_gpu_iqr.py                 # GPU total-runtime IQR band (depths 1-13)
python scripts/analysis/extract_cpu_iqr.py                 # CPU total-runtime IQR band (depths 1-7)
python scripts/analysis/extract_eshn_iqr.py                # ES-HyperNEAT (port) baseline band
python scripts/analysis/extract_gpu_iqr_extended.py        # GPU runtime broken down by population
python scripts/analysis/extract_gpu_jit_times.py           # GPU JIT/construction overhead per depth
python scripts/analysis/extract_gpu_jit_times_extended.py  # GPU JIT overhead by population
python scripts/analysis/extract_depth_8_11_iqr.py          # deep-substrate detail (depths 8-11)
python scripts/analysis/extract_depth_10_13_pop1000_memmap.py  # deep-substrate tail (depths 8-13)
python scripts/analysis/compute_speedup.py                 # EMR-vs-ES-HN total-runtime speedup
python scripts/analysis/compute_ccm_runtime.py             # CCM financial-dataset runtime (ES-HN vs EMR)
python scripts/analysis/analyze_benchmark_data.py          # per-generation scaling summary + LaTeX table

# Figures: the two poster figures are standalone TikZ (scripts/figures/*.tex) ->
# compile to figures/<name>.{pdf,png}
for f in fig_lazy_vs_eager fig_total_runtime; do
  ( cd scripts/figures && pdflatex -interaction=nonstopmode "$f.tex" && mv -f "$f.pdf" ../../figures/ && rm -f "$f".{aux,log} )
  pdftoppm -png -r 300 -singlefile "figures/$f.pdf" "figures/$f"
done
```

The paper reports the speedups with the settings they were measured under.

### Data the scripts consume (in `data/`, from the data release)

Each `runs_*.json` holds a `runs` list; every run records `depth`, `population`, `seed`,
`n_generations` (30), `total_runtime_s` (construction plus the sum of `per_gen_times_s`),
`per_gen_times_s` (30 per-generation times, see the timing provenance below),
`steady_per_gen_s`, `construction_overhead_s` (one-time JIT/construction),
`construction_in_gen0` (true where generation 0's time includes the construction and
`construction_overhead_s` is 0: the CCM EMR series), and `in_figure_band`.

**Timing provenance.** Release 0.1.0's README called every runtime number a measured
30-generation total with nothing projected from shorter runs. Each `total_runtime_s` is
`construction_overhead_s` plus the sum of `per_gen_times_s`, and no `per_gen_times_s` series is
a measurement: each holds lognormal draws scaled so that their mean is `steady_per_gen_s` (the
CCM EMR series differ, see below). A total is therefore the construction time plus 30 times the
steady time. Those two times are measured except as this list says; it also names the records
without a seed and the runs projected from shorter ones.

- `runs_emr_gpu.json`: depths 1 to 7 hold one value per population, without a seed. At depth 7,
  population 750 is interpolated and population 1000's 60 s steady time is an estimate. The runs
  at depths 8 to 13 lasted 10 generations (6 for population 300, seed 123, which solved XOR at
  depths 8 to 12), so their totals project 30; depth 13's population-1000 seed 456 was not run,
  and its times are depth 12's, scaled by 11.0549 (construction) and 11.9349 (steady).
- `runs_emr_cpu.json`: the 48 runs at depths 8 and 9 lasted 10 generations, so their totals
  project 30, and they were timed on a later version of the code, about 1.2 times slower than
  the one behind depths 1 to 7.
- `runs_eshn.json`: depths 1 to 6 hold one value per population, averaged over its seeds. At
  depth 7 the construction time is a 0.04 s constant in all 24 runs, not a measurement.
- `runs_ccm.json`: one record per method and depth, without a seed, from runs of 5 generations
  on a 5,000-observation subset of the 94,464-observation panel. ES-HyperNEAT's construction is
  0 by convention. Each EMR series holds the JIT time in generation 0 and 29 steady generations
  whose mean is `steady_per_gen_s`, so its total is generation 0 plus 29 times the steady time
  (138.7 s at depth 1, where 30 times the steady time would give 132 s).

The scripts' printed headers still say "real 30-generation runs"; this note supersedes them.

| File | Used by |
|------|---------|
| `runs_emr_gpu.json` | `extract_gpu_*.py`, `extract_depth_*.py`, `compute_speedup.py`, `analyze_benchmark_data.py` |
| `runs_emr_cpu.json` | `extract_cpu_iqr.py`, `compute_speedup.py` |
| `runs_eshn.json` | `extract_eshn_iqr.py`, `compute_speedup.py`, `analyze_benchmark_data.py` |
| `runs_ccm.json` | `compute_ccm_runtime.py` |
| `gecco2026_*.json`, `statistical_validation_results.json` | supplementary preset/depth/multihop run outputs |

## Re-running the experiments from scratch (optional)

The timing benchmarks were measured on specific hardware (Apple Silicon M4 Max CPU; a
single NVIDIA RTX 2080 Ti 11 GB GPU). The from-scratch runners `scripts/runners/run_gecco2026_experiments.py`
and `scripts/runners/run_statistical_validation.py` were produced on the original eager-multi-resolution
implementation, the HMR engine, which release 0.1.0 ships; their published files
(`gecco2026_*.json`, `statistical_validation_results.json`) reproduce on 0.1.0 with
`python scripts/replicate.py run emr-hyperneat --release 0.1.0 -- <command>` from the repository
root. From 0.2.0 both run EMR-HyperNEAT, as the experiments ran, on EMR's defaults except
HMR's level-0 variance rule and random key split, which `scripts/runners/emr_settings.py` pins (set
to False, it runs EMR's defaults): the hidden-to-hidden cache is off except in the caching
ablation's cache arms, and the multi-hop expansion composes each genome's own connections.
`with_lateral` and `with_self` run the connection classes their names say, which departs from what
the HMR engine ran: it ran both as hidden_only, and the published files hold those runs, which 0.1.0
reproduces. `run_statistical_validation.py`'s single run is `run_xor_experiment`
(`run_hmr_xor_experiment` in 0.1.0). They write to `results/gecco2026_emr/` and
`results/statistical_validation_emr/`, apart from `data/`. A GPU is needed to reproduce the timing.
Regenerating the tables/figures above does not require them.

At the validation configuration, one seed per configuration (42 where the row names none) with full
generation budgets, both engines on one x86 server (two Intel Xeon E5-2683 v4, one CPU per run,
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, jax 0.6.1), EMR in release 0.2.0 gives the
right-hand column where the HMR engine of release 0.1.0 gave the left. A cell is the generation of
the first solve, or not solved, with the best fitness; one seed shows where the two engines agree
and differ, not a solve rate. The statistical validation gives the same generations and best fitness
on both engines, seed by seed (feed-forward evaluation, with HMR's level-0 rule and key split
pinned). The HMR engine's with_lateral and with_self runs equal its hidden_only run, as it ran both
as hidden_only; EMR runs their classes.

| Run | HMR engine (release 0.1.0) | EMR (release 0.2.0) |
|---|---|---|
| `run_gecco2026_experiments.py`, preset feedforward, depth 4 | solved at 24 (0.989361) | solved at 12 (0.996287) |
| `run_gecco2026_experiments.py`, preset hidden_only, depth 4 | solved at 1 (0.999985) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py`, preset with_backward, depth 4 | solved at 1 (1.000000) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py`, preset with_lateral, depth 4 | solved at 1 (0.999985) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py`, preset with_self, depth 4 | solved at 1 (0.999985) | solved at 3 (1.000000) |
| `run_gecco2026_experiments.py`, preset full_recurrent, depth 4 | solved at 1 (0.999998) | solved at 1 (0.999998) |
| `run_gecco2026_experiments.py --experiment multihop`, iteration level 1 | solved at 1 (0.996215) | solved at 1 (0.996215) |
| `run_gecco2026_experiments.py --experiment multihop`, iteration level 2 | solved at 1 (0.999985) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py --experiment multihop`, iteration level 3 | solved at 1 (1.000000) | solved at 2 (0.995264) |
| `run_gecco2026_experiments.py --experiment multihop`, iteration level 4 | solved at 4 (0.999996) | solved at 3 (1.000000) |
| `run_gecco2026_experiments.py --experiment depth`, depth 3 | solved at 1 (1.000000) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py --experiment depth`, depth 4 | solved at 1 (0.999985) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py --experiment depth`, depth 5 | solved at 1 (1.000000) | solved at 1 (1.000000) |
| `run_gecco2026_experiments.py --experiment depth`, depth 6 | solved at 1 (0.999986) | solved at 1 (1.000000) |
| `run_statistical_validation.py --depths 3 --seeds 5` | 4 of 5 solved (at 2, 3, 1, 2); seed 43 not solved (0.874981) | 4 of 5 solved (at 2, 3, 1, 2); seed 43 not solved (0.874981) |
| `run_statistical_validation.py --depths 4 --seeds 5` | 4 of 5 solved (at 2, 3, 1, 2); seed 43 not solved (0.874981) | 4 of 5 solved (at 2, 3, 1, 2); seed 43 not solved (0.874981) |

## Citation

PDF: [claret.tech/pdf/claret2026emr](https://claret.tech/pdf/claret2026emr) ·
DOI: [10.1145/3795101.3805361](https://doi.org/10.1145/3795101.3805361)

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
