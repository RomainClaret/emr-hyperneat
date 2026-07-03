# Per-Node Activation Function Evolution

**Per-Node Activation Function Evolution in Indirectly Encoded Substrates**, ALIFE 2026.

> Publication: ALIFE 2026, DOI forthcoming. BibTeX in [Citation](#citation) below.
> The paper PDF is not included.

This paper adds **per-node activation function evolution** to EMR-HyperNEAT: instead of a
single fixed activation, each node's activation is selected from an 18-function palette,
either globally, from a CPPN output channel, or from connection-weight interpretation. The
central result is the **oscillatory/monotonic divide**: oscillatory activations (sin and
relatives) solve parity/XOR-structured problems that every monotonic activation fails at.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for which experiment runs on which
implementation (EMR vs the frozen HMR module) and why. It is the authoritative guide.

## Experiments

| ID | Runner | Implementation |
|----|--------|----------------|
| E1 | `scripts/runners/E1_parity_scaling.py` | EMR (`extra_randkey_split=True`, `pop_size=150`) |
| E2 | `scripts/runners/E2_xor_per_function.py` | EMR |
| E6 | `scripts/runners/E6_gradient_baseline.py` | plain JAX MLP (no HyperNEAT) |
| E7 | `scripts/runners/E7_two_spirals_per_function.py` | EMR |
| E8 | `scripts/runners/E8_depth_sensitivity.py` | EMR |
| E9 | `scripts/runners/E9_lts_low_extended.py` | EMR |
| E3 | population sensitivity (`cppn_output`) | **HMR** (frozen module, in `_hmr_frozen`) |
| E4 / E5 | recurrence | **HMR** (frozen module) |

The EMR runners (E1/E2/E7/E8/E9) bit-reproduce the published per-seed trajectories. E3/E4/E5
rely on the `cppn_output`/recurrence forward passes that EMR reimplements differently, so they
are reproduced on the frozen HMR module (see `REPRODUCIBILITY.md`).

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

DOI: _to appear_

```bibtex
@inproceedings{claret2026activations,
  title={Per-Node Activation Function Evolution in Indirectly Encoded Substrates: Solvability, Limits, and Emergent Diversity},
  author={Claret, Romain and O'Neill, Michael and Cotofrei, Paul and Stoffel, Kilian},
  booktitle={Artificial Life Conference Proceedings 38},
  year={2026},
  publisher={MIT Press},
}
```
