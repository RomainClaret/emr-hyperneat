# Bio-Inspired Palette Evolution

**Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility
Shapes Activation Function Discovery**, PPSN 2026.

> Paper: **[PDF](https://claret.tech/pdf/claret2026bio)** · DOI [10.1007/978-3-032-36217-9_23](https://doi.org/10.1007/978-3-032-36217-9_23) (PPSN 2026). BibTeX in [Citation](#citation) below.
> The PDF is hosted at claret.tech, not committed to this repository.
> Code release: **emr-hyperneat 0.1.0** (tag [`v0.1.0`](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions-bio-inspired)); run a command below on it with `python scripts/replicate.py run emr-dynamic-functions-bio-inspired -- <command>` from the repository root ([choosing a release](../../docs/reproducing-experiments.md#choosing-a-release)).

Where the per-node activation paper showed *that* oscillatory activations matter, this paper
asks *how evolution discovers them*. It evaluates a library of **bio-inspired palette-evolution
strategies** that meta-evolve which activation (and aggregation) functions are available to the
substrate over generations: temporal-credit (STDP/Hebbian), oscillatory gating (circadian/burst),
immune memory (clonal selection), developmental (critical periods/neurogenesis), ecological
(predator-prey/succession), and homeostatic (BCM/fatigue). The central
finding is a correlation: strategies whose **timescale** matches the evolutionary search
horizon reach higher solve rates (a rank correlation over 11 strategies). The rescaling
experiment's GRN arm, the paper's causal evidence, does not show causation: the rescaled GRN's
activation palette ends at the same two functions in 30 of 30 seeds, whatever the fitness.

## The method: `scripts/palette_strategies/`

The strategy library (framework-independent) is the paper's core contribution: a
`PaletteEvolutionStrategy` base class (`initialize` / `get_active_palette` /
`post_generation_update`) with ~46 concrete bio-inspired strategies across the six categories
above, plus dual (activation+aggregation) variants. These are passed to the substrate
algorithm, which queries the active palette each generation.

## Reproducing the paper's tables and figures

The analysis and figure scripts regenerate the paper's quantitative content **from the
result data release** (fetch the data release first; it populates `results/`, 44 experiment
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

See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) for the full benchmark -> runner -> result-dir
map.

## Re-running the experiments

From release 0.2.0 the 10 experiment runners (`scripts/runners/benchmark_bio_*.py`) apply the
palette strategies on EMR-HyperNEAT. The published runs used the HMR engine (its aggregation
variant, and its full variant for `benchmark_bio_topology_full.py`), which release 0.1.0
ships; from the repository root, a runner's published version runs on 0.1.0 with

```bash
python scripts/replicate.py run emr-dynamic-functions-bio-inspired --release 0.1.0 -- \
    python scripts/runners/benchmark_bio_grn_p5.py --pilot
```

Each EMR version runs its experiment as it ran, on EMR's defaults except where its docstring says:
population 150 where the HMR variant ignored the population argument (the argument is kept and
ignored), iteration level 2 for the recurrent topology runs, and HMR's level-0 variance rule and
random key split, which `scripts/runners/emr_settings.py` pins in every runner (set to False, it
runs EMR's defaults). Its docstring says what differs from the published runs, and it writes to
`results/<dir>_emr/`, apart from the published runs in `results/<dir>/`. Run e.g.
`python scripts/runners/benchmark_bio_grn_p5.py`; the analysis and figure regeneration above does
not require them.

At the validation configuration, one seed per configuration (42 where the row names none) with full
generation budgets, both engines on one x86 server (two Intel Xeon E5-2683 v4, one CPU per run,
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, jax 0.6.1), EMR in release 0.2.0 gives the
right-hand column where the HMR engine of release 0.1.0 gave the left. A cell is the generation of
the first solve, or not solved, with the best fitness; one seed shows where the two engines agree
and differ, not a solve rate. Every run differs from the published one: EMR reads each node's
function channels at its self-connection, where HMR averaged them over the node's input queries. The
two population labels of `benchmark_bio_pop_sensitivity_p5.py` give one run on each engine, as both
run at population 150.

| Run | HMR engine (release 0.1.0) | EMR (release 0.2.0) |
|---|---|---|
| `benchmark_bio_agg_oracle.py`, TestAggregationOraclePaper's call (15 generations) | not solved (0.767838) | not solved (0.765859) |
| `benchmark_bio_agg_oracle.py --condition monotonic_min` | not solved (0.780973) | not solved (0.766666) |
| `benchmark_bio_agg_oracle.py --condition bandpass_integrate_min` | not solved (0.766666) | not solved (0.767839) |
| `benchmark_bio_grn_p5.py` | not solved (0.766094) | solved at 148 (0.920417) |
| `benchmark_bio_grn_p6p7.py --problem parity_6` | solved at 121 (0.866023) | solved at 31 (0.879111) |
| `benchmark_bio_grn_p6p7.py --problem parity_7` | not solved (0.751953) | solved at 75 (0.904635) |
| `benchmark_bio_nonparity_cl.py --sequence forward --strategy circadian_rhythm_dual` | parity_4 at 19 (0.976366); two_moons at 23 (0.928519); parity_4 at 27 (0.991078) | parity_4 at 11 (0.977746); two_moons at 52 (0.909507); parity_4 at 14 (0.987914) |
| `benchmark_bio_p4_n60_extension.py --strategy circadian_rhythm_dual` (seed 72) | solved at 100 (0.998199) | solved at 79 (0.954523) |
| `benchmark_bio_pop_sensitivity_p5.py --strategy circadian_rhythm_dual --pop-size 200` | solved at 58 (0.961932) | solved at 17 (0.960746) |
| `benchmark_bio_pop_sensitivity_p5.py --strategy circadian_rhythm_dual --pop-size 800` | solved at 58 (0.961932) | solved at 17 (0.960746) |
| `benchmark_bio_sine_regression.py --strategy circadian_rhythm_dual` | solved at 2 (0.996526) | solved at 2 (0.996501) |
| `benchmark_bio_strategy_fixed_palette.py`, TestBioInspiredPaper's call (6 generations) | not solved (0.766612) | not solved (0.775485) |
| `benchmark_bio_strategy_fixed_palette.py --strategy circadian_rhythm_dual` | solved at 24 (0.983718) | solved at 41 (0.969574) |
| `benchmark_bio_timescale_rescaling.py --strategy grn_rescaled` | not solved (0.887958) | not solved (0.939799) |
| `benchmark_bio_timescale_rescaling.py --strategy glial_rescaled` | not solved (0.863402) | not solved (0.901056) |
| `benchmark_bio_timescale_rescaling.py --strategy ant_colony_rescaled` | solved at 16 (0.999920) | solved at 22 (1.000000) |
| `benchmark_bio_topology_full.py --strategy circadian_rhythm_dual` | solved at 2 (1.000000) | solved at 9 (1.000000) |

## Shared infrastructure

`scripts/classification_problems.py` is duplicated into this paper. Problems needing the full
geenns benchmark suite (Retina, Multiplexer) raise `NotImplementedError` here.

## Citation

PDF: [claret.tech/pdf/claret2026bio](https://claret.tech/pdf/claret2026bio) ·
DOI: [10.1007/978-3-032-36217-9_23](https://doi.org/10.1007/978-3-032-36217-9_23)

```bibtex
@inproceedings{claret2026bio,
  title={Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility Shapes Activation Function Discovery},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={Parallel Problem Solving from Nature -- PPSN XIX},
  year={2026},
  organization={Springer},
  doi={10.1007/978-3-032-36217-9_23},
  url={https://claret.tech/pdf/claret2026bio}
}
```
