# Changelog

Notable changes to the released `emr_hyperneat` package. The version tracks
`emr_hyperneat/_version.py`.

## 0.1.0

Initial public release of EMR-HyperNEAT as a standalone, installable package.

- **Engine**: Eager Multi-Resolution HyperNEAT — eager tensor evaluation on a pre-computed
  hierarchical grid plus post-hoc variance masking, replacing ES-HyperNEAT's sequential quadtree.
  Execution modes `auto | standard | streaming | depth_offload` trade peak memory for speed and
  reach depth 11+ on a single device.
- **Evolvable computation**: per-node activation and aggregation functions, palette evolution, and
  4-vector neuromodulation for multi-behavior genotypes.
- **Substrate rule**: `variance_rule` defaults to `canonical` (one substrate across execution
  modes); `legacy` reproduces the published standard-path numbers, and the frozen `_hmr_frozen`
  module reproduces specific published tables bit-for-bit.
- **Multi-GPU**: single-device by default; `full_pipeline_parallel`, `eval_only_parallel`, and
  `streaming` distribute across devices through `run_until_threshold`.
- **Tests**: cross-layout contracts assert bitwise on CPU and to float32 rounding on a GPU;
  golden-value regressions on the pinned JAX; paper-validation tests re-run each paper's simplest
  experiment. The CPU suite runs under `pytest -n 4 --dist loadfile`.
- **Reproducibility**: the accompanying papers ship their runners; result data is a separate Zenodo
  release (DOI 10.5281/zenodo.21383893), fetched not committed.
