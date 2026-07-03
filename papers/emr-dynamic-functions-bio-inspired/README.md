# Bio-Inspired Palette Evolution

**Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility
Shapes Activation Function Discovery**, PPSN 2026.

> Publication: PPSN 2026, DOI forthcoming. BibTeX in [Citation](#citation) below.
> The paper PDF is not included.

Where the per-node activation paper showed *that* oscillatory activations matter, this paper
asks *how evolution discovers them*. It evaluates a library of **bio-inspired palette-evolution
strategies** that meta-evolve which activation (and aggregation) functions are available to the
substrate over generations: temporal-credit (STDP/Hebbian), oscillatory gating (circadian/burst),
immune memory (clonal selection), developmental (critical periods/neurogenesis), ecological
(predator-prey/succession), and homeostatic (BCM/fatigue). The central
finding is that a strategy's **timescale** must match the evolutionary search horizon for it to
discover the right functions.

## The method: `scripts/palette_strategies/`

The strategy library (framework-independent) is the paper's core contribution: a
`PaletteEvolutionStrategy` base class (`initialize` / `get_active_palette` /
`post_generation_update`) with ~46 concrete bio-inspired strategies across the six categories
above, plus dual (activation+aggregation) variants. These are passed to the substrate
algorithm, which queries the active palette each generation.

## Reproducing the paper's tables and figures

The analysis and figure scripts regenerate the paper's quantitative content **from the
result data release** (fetch the data release first; it populates `results/`, 45 experiment
subdirectories):

```bash
python scripts/analysis/analysis_paper2.py                 # main per-strategy statistics
python scripts/analysis/analysis_strengthening.py          # strengthening-round summary
python scripts/analysis/analysis_oscillatory_composition.py
python scripts/analysis/analysis_timescale_correlation.py  # timescale-vs-discovery correlation
python scripts/analysis/analyze_grn_mechanism.py           # GRN-rescaled non-oscillatory pathway
python scripts/analysis/analyze_palette_overlap.py
python scripts/analysis/verify_paper_numbers.py            # cross-check reported numbers

# figures -> figures/*.{pdf,png}
python scripts/figures/generate_fig7_discovery_timing.py   # Fig 1 (reads results/single_task/)
python scripts/figures/generate_fig2_pareto.py             # Fig 2 (reads results/stdp_hebbian_replication*/)
```

> `generate_fig2_pareto.py` needs the `stdp_hebbian_replication*` single-task results; it skips with
> a message if that directory is not present in the fetched data.

See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) for the full workstream → runner → result-dir
map and completion status.

## Re-running the experiments (requires the frozen HMR module)

The 10 experiment runners (`scripts/runners/benchmark_bio_*.py`) apply the palette strategies through the
original eager-multi-resolution implementation (the `hmrhyperneat` modules
`*_dynamic_functions_aggregation` and `*_unified_extended_dynamic_functions_full`), vendored as
the frozen HMR module under `emr_hyperneat/_hmr_frozen/`. Run e.g.
`python scripts/runners/benchmark_bio_grn_p5.py`; the analysis and figure regeneration above does not
require them.

## Shared infrastructure

`scripts/classification_problems.py` is duplicated into this paper. Problems needing the full
geenns benchmark suite (Retina, Multiplexer) raise `NotImplementedError` here.

## Citation

DOI: _to appear_

```bibtex
@inproceedings{claret2026bio,
  title={Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility Shapes Activation Function Discovery},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={International Conference on Parallel Problem Solving from Nature},
  year={2026},
  organization={Springer}
}
```
