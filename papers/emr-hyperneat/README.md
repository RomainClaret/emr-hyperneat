# EMR-HyperNEAT (the algorithm paper)

**Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates**
Romain Claret, Michael O'Neill, Paul Cotofrei, Kilian Stoffel, GECCO 2026.

> Publication: DOI [10.1145/3795101.3805361](https://doi.org/10.1145/3795101.3805361) (GECCO Companion '26). BibTeX in [Citation](#citation) below.
> The paper PDF is not included in this repository.

This is the paper that introduces **EMR-HyperNEAT** itself: replacing ES-HyperNEAT's
sequential, CPU-bound quadtree with eager tensor evaluation on a pre-computed
hierarchical grid plus variance masking, so substrate discovery `vmap`s across a whole
population on the GPU. It reports the resulting speedups over ES-HyperNEAT (PUREPLES)
across substrate depths and population sizes, the depth/recurrence-preset studies, and
the JIT-vs-steady-state timing breakdown.

## What this paper adds to the algorithm

The base `emr_hyperneat.EMRHyperNEAT`: hierarchical-grid substrate discovery,
variance-based masking, batched CPPN queries, and the two-level (JAX compilation + h→h
connection) caching. Everything the later papers build on.

## Reproducing the paper's tables and figures

The paper's quantitative content is regenerated **from the measurement data**:
no GPU and no evolution run required. Every runtime number is a measured 30-generation
`total_runtime_s`; nothing is projected from shorter runs. Fetch the data release first
(see root `scripts/fetch_results.py`), which populates `data/`. Then:

```bash
# Runtime IQR bands + speedups, from the real 30-generation runs (scripts/analysis/):
python scripts/analysis/extract_gpu_iqr.py                 # GPU total-runtime IQR band (depths 1-13)
python scripts/analysis/extract_cpu_iqr.py                 # CPU total-runtime IQR band (depths 1-7)
python scripts/analysis/extract_eshn_iqr.py                # ES-HyperNEAT (PUREPLES) baseline band
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

Headline: at depth 7 (pop=1000) EMR-HyperNEAT is **~34x faster per generation** than
ES-HyperNEAT, and its **total-runtime speedup over 30 generations is ~59-81x** (IQR
midpoint to true-median ratio); on CPU the total-runtime speedup is ~11x.

### Data the scripts consume (in `data/`, from the data release)

Each `runs_*.json` holds a `runs` list; every run records `depth`, `population`, `seed`,
`n_generations` (30), the measured `total_runtime_s`, `steady_per_gen_s`,
`construction_overhead_s` (one-time JIT/construction), and `in_figure_band`.

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
implementation and run on the frozen HMR module (`emr_hyperneat/_hmr_frozen/`); a GPU is needed to
reproduce the timing. Regenerating the tables/figures above does not require them.

## Citation

DOI: [10.1145/3795101.3805361](https://doi.org/10.1145/3795101.3805361)

```bibtex
@inproceedings{claret2026emr,
  title={Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO Companion '26)},
  year={2026},
  address={San Jose, Costa Rica},
  publisher={ACM},
  doi={10.1145/3795101.3805361}
}
```
