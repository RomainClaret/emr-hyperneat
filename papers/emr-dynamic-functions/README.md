# Per-Node Activation Function Evolution

**Per-Node Activation Function Evolution in Indirectly Encoded Substrates**, ALIFE 2026.

> Paper: **[PDF](https://claret.tech/pdf/claret2026activations)** · DOI [10.1162/ISAL.a.1042](https://doi.org/10.1162/ISAL.a.1042) (ALIFE 2026). BibTeX in [Citation](#citation) below.
> The PDF is hosted at claret.tech, not committed to this repository.
> Code release: **emr-hyperneat 0.1.0** (tag [`v0.1.0`](https://github.com/RomainClaret/emr-hyperneat/tree/v0.1.0/papers/emr-dynamic-functions)); run a command below on it with `python scripts/replicate.py run emr-dynamic-functions -- <command>` from the repository root ([choosing a release](../../docs/reproducing-experiments.md#choosing-a-release)).

This paper adds **per-node activation function evolution** to EMR-HyperNEAT: instead of a
single fixed activation, each node's activation is selected from an 18-function palette,
either globally, from a CPPN output channel, or from connection-weight interpretation. The
central result is the **oscillatory/monotonic divide**: oscillatory activations (sin and
relatives) solve feed-forward parity beyond XOR, which monotonic activations almost never do
(Parity-4: sin 30 of 30 runs, nine of the ten monotonic functions 0 of 30).

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for which implementation each experiment's
published runs used (EMR or the HMR engine) and how they reproduce. It is the authoritative guide.

## Experiments

| ID | Runner | Implementation |
|----|--------|----------------|
| E1 | `scripts/runners/E1_parity_scaling.py` | EMR (`extra_randkey_split=True`, `pop_size=150`) |
| E2 | `scripts/runners/E2_xor_per_function.py` | EMR |
| E6 | `scripts/runners/E6_gradient_baseline.py` | plain JAX MLP (no HyperNEAT) |
| E7 | `scripts/runners/E7_two_spirals_per_function.py` | EMR |
| E8 | `scripts/runners/E8_depth_sensitivity.py` | EMR |
| E9 | `scripts/runners/E9_lts_low_extended.py` | EMR |
| E3 | `scripts/runners/E3_population_sensitivity.py` | EMR from 0.2.0; published runs: **HMR** |
| E4 / E5 | `scripts/runners/E4_recurrent_parity8.py`, `scripts/runners/E5_recurrence_type.py` | EMR from 0.2.0; published runs: **HMR** |

The EMR runners (E1/E2/E7/E8/E9) bit-reproduce the published per-seed trajectories. E3/E4/E5
were published on the HMR engine, whose `cppn_output` and recurrent evaluations EMR computes
differently. From 0.2.0 they are EMR versions that run each experiment as it ran on EMR's defaults,
with HMR's level-0 variance rule and random key split pinned (`scripts/runners/emr_settings.py`), as
E1/E2/E7/E8/E9 pin them, and write to `results/<experiment>_emr/` (E3 to the bio-inspired paper's
`results/population_sensitivity_emr/`, next to its published runs); the published runs reproduce
on release 0.1.0 with
`python scripts/replicate.py run emr-dynamic-functions --release 0.1.0 -- <command>` (see
`REPRODUCIBILITY.md`).

At the validation configuration, one seed per configuration (42 where the row names none) with full
generation budgets, both engines on one x86 server (two Intel Xeon E5-2683 v4, one CPU per run,
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, jax 0.6.1), EMR in release 0.2.0 gives the
right-hand column where the HMR engine of release 0.1.0 gave the left. A cell is the generation of
the first solve, or not solved, with the best fitness; one seed shows where the two engines agree
and differ, not a solve rate. E4 and E5 ran 100 generations here (their scripts allow 300). The
recurrent runs differ most: in 100 generations EMR solves four of E5's five recurrent presets (HMR:
hidden_only alone) and E4's tanh (HMR: not solved); E3's two population labels give one run on each
engine.

| Run | HMR engine (release 0.1.0) | EMR (release 0.2.0) |
|---|---|---|
| `E3_population_sensitivity.py --strategy circadian_rhythm_dual --pop_size 200` | not solved (0.870575) | solved at 13 (1.000000) |
| `E3_population_sensitivity.py --strategy circadian_rhythm_dual --pop_size 1000` | not solved (0.870575) | solved at 13 (1.000000) |
| `E4_recurrent_parity8.py`, sin (100 generations) | solved at 3 (0.996094) | solved at 2 (0.995908) |
| `E4_recurrent_parity8.py`, tanh (100 generations) | not solved (0.874561) | solved at 57 (0.994582) |
| `E5_recurrence_type.py`, feedforward (100 generations) | not solved (0.870243) | not solved (0.901229) |
| `E5_recurrence_type.py`, hidden_only (100 generations) | solved at 4 (1.000000) | solved at 49 (1.000000) |
| `E5_recurrence_type.py`, with_backward (100 generations) | not solved (0.892748) | solved at 18 (0.999999) |
| `E5_recurrence_type.py`, with_lateral (100 generations) | not solved (0.870383) | solved at 8 (0.999688) |
| `E5_recurrence_type.py`, with_self (100 generations) | not solved (0.937500) | not solved (0.945885) |
| `E5_recurrence_type.py`, full_recurrent (100 generations) | not solved (0.905124) | solved at 78 (0.950597) |

## Running

```bash
# install per the root README (pip install -e . ; pip install -e third_party/tensorneat)
JAX_PLATFORMS=cpu python scripts/runners/E2_xor_per_function.py --function sin --seeds 30 --max-gens 300
```

Each runner writes to this paper's `results/<experiment>_n30/` (paper-local, resolved from
`__file__`). The full result set ships via the external data release (root
`scripts/fetch_results.py`).

## Analysis and figures

The `scripts/analysis/analysis_*.py` scripts read `results/` and recompute the paper's
statistics:

```bash
python scripts/analysis/analysis_paper_statistics.py     # every reported per-function statistic
python scripts/analysis/analysis_extrema_correlation.py  # local-extrema count vs solve rate
python scripts/analysis/analysis_plateau_fitness.py      # fitness plateau dynamics
python scripts/analysis/analysis_strengthening.py        # strengthening-round summary
```

`scripts/figures/generate_fig_system_overview.py` and `scripts/figures/generate_fig_zero_crossing.py`
regenerate the paper's two figures into `figures/*.{pdf,png}`; both are self-contained (no data needed).

## Shared infrastructure

`scripts/classification_problems.py` is duplicated into this paper (synthetic Parity / XOR /
Two-Spirals problems used by the runners). Problems that require the full geenns benchmark
suite (Retina, Multiplexer) raise `NotImplementedError` here. They are not part of this paper.

## Citation

PDF: [claret.tech/pdf/claret2026activations](https://claret.tech/pdf/claret2026activations) ·
DOI: [10.1162/ISAL.a.1042](https://doi.org/10.1162/ISAL.a.1042)

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
```

> **Reproduction note:** the published runs used the legacy level-0 subdivision
> rule. The EMR runners here pin `variance_rule: legacy` so they still reproduce the paper's
> numbers bit for bit; the package default is `canonical` (see `docs/configuration.md`).
