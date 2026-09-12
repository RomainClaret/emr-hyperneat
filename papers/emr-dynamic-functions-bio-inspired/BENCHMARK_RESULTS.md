# Bio-Inspired Palette Evolution: N=30 Benchmark Results

Benchmark results for the PPSN 2026 paper *"Bio-Inspired Palette Evolution in
Indirectly Encoded Substrates"*.

> **Note:** the `scripts/benchmarks/emr_optimization/...` script paths named below are from the
> original research monorepo and are not shipped in this standalone repository; the released
> runners live under `scripts/runners/`. Result data ships via the data release (root
> `scripts/fetch_results.py`).

## Experiment Tracking

### Workstream A: Single-Task 8-Strategy Comparison (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/stdp_hebbian_replication.py`
- **Covers claims**: #1-4 (Table 1)
- **Runs**: 240 experiments
- **Output dir**: `scripts/benchmarks/emr_optimization/results/stdp_hebbian_replication_20260206_163126/`

**Results**:

| Strategy | Solve Rate | 95% CI Wilson | Med Gen | Sin Disc. |
|---|---|---|---|---|
| **Circadian** | **97%** | [0.833, 0.994] | **20** | 93% |
| Hebbian | 90% | [0.744, 0.965] | 16 | 100% |
| Crit. Period | 90% | [0.744, 0.965] | 31 | 100% |
| STDP | 90% | [0.744, 0.965] | 34 | 100% |
| Metaplastic | 80% | [0.627, 0.905] | 24 | 100% |
| Baseline | 70% | [0.521, 0.833] | 59 | 37% |
| Pred.-Prey | 67% | [0.488, 0.808] | 69 | 0% |
| Neurogenesis | 53% | [0.361, 0.698] | 54 | 37% |

**Statistics**: Kruskal-Wallis H=54.16, p=2.19×10⁻⁹

### Workstream B: Continual Learning, Elite Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/run_cl_elite_sequential.sh`
- **Covers claims**: #6-8 (Table 2)
- **Runs**: 1,470 experiments = 7 strategies × 30 seeds × 7 tasks
- **Output dir**: `results/continual_learning_elite_n30/`

**Results**:

| Strategy | Solve% | Total/7 | AA | Sin Ret.% | Sin Disc.% | Palette Stab.% |
|---|---|---|---|---|---|---|
| sin_default_dual | 70.0 | 4.9 | 0.835 | **100.0** | 100 | 100.0 |
| critical_period_refined | 70.0 | 4.9 | 0.835 | 74.3 | 100 | 9.7 |
| circadian_rhythm | 70.0 | 4.9 | 0.835 | 54.2 | 97 | 22.1 |
| baseline | 70.0 | 4.9 | 0.835 | 46.4 | 93 | 25.7 |
| stdp | 70.0 | 4.9 | 0.835 | 45.3 | 80 | 19.9 |
| consolidation_window | 70.0 | 4.9 | 0.835 | 41.2 | 73 | 47.0 |
| clonal_selection | 70.0 | 4.9 | 0.835 | **0.0** | 0 | 94.4 |

**Per-Task Solve Rates** (identical across all strategies):
XOR=50%, Parity-3=37%, Parity-4=60%, Parity-5=80%, Parity-6=63%, Parity-7=100%, Parity-8=100%

**Critical Finding**: All strategies produce **identical** solve/fail patterns per seed.
Strategies affect palette composition (sin retention) but NOT evolutionary trajectory.
NEAT seed determinism overrides strategy effects on topology evolution.
FWT = +85.9% across all strategies (later parity tasks trivially solved after earlier ones).

### Workstream C: Continual Learning, Hybrid Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/run_cl_hybrid_sequential.sh`
- **Covers claims**: #11-12 (FWT/BWT)
- **Runs**: 1,050 experiments = 5 strategies × 30 seeds × 7 tasks
- **Output dir**: `results/continual_learning_hybrid_n30/`

**Results**:

| Strategy | Solve% | Total/7 | Sin Ret.% |
|---|---|---|---|
| critical_stdp | 70.0 | 4.9 | 48.9% |
| circadian_critical | 70.0 | 4.9 | 46.9% |
| stdp_consolidation | 70.0 | 4.9 | 43.8% |
| circadian_clonal | 70.0 | 4.9 | 0.0% |
| consolidation_clonal | 70.0 | 4.9 | 0.0% |

**Confirms elite finding**: identical solve rates across all strategies.
Clonal selection hybrids inherit 0% sin retention from their clonal component.

### Workstream D: Dual-Domain Bio Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_bio_dual_palette.py`
- **Covers claims**: #13 (dual-domain discovery)
- **Runs**: 150 experiments
- **Output dir**: `results/bio_dual_palette_20260206_163127/`

**Results**:

| Strategy | Solve% | Sin Disc.% | Sin Ret.% | Agg Disc.% |
|---|---|---|---|---|
| **Clonal Selection** | **90%** | 3% | 0% | 3% |
| Circadian | 87% | 93% | 50% | 0% |
| STDP | 77% | 67% | 37% | 50% |
| Baseline | 77% | 43% | 37% | 0% |
| Crit. Period Refined | 77% | 50% | 13% | 0% |

### Workstream E: Failed Mechanisms (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_bio_dual_palette.py`
- **Covers claims**: #14-16 (failed mechanisms)
- **Runs**: 90 experiments
- **Output dir**: `results/bio_dual_palette_20260206_163130/`

**Results**:

| Mechanism | Solve Rate | 95% CI | Root Cause |
|---|---|---|---|
| GRN | 3% | [0.01, 0.17] | Expression dynamics too slow |
| Glial Modulation | 47% | [0.30, 0.64] | Modulatory timescale exceeds eval |
| Ant Colony Pheromone | 80% | [0.63, 0.90] | Pheromone accumulation lag |

---

## Summary Statistics

| Workstream | Experiments | Strategies |
|---|---|---|
| A (Single-Task) | 240 | 8 |
| B (CL Elite) | 1,470 | 7 |
| C (CL Hybrid) | 1,050 | 5 |
| D (Dual-Domain) | 150 | 5 |
| E (Failed) | 90 | 3 |
| **Total (A-E)** | **3,000** |— |

---

### Exp 3: Tier 2 Replication (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_tier2_replication.py`
- **Analysis**: `scripts/benchmarks/emr_optimization/analysis_tier2_replication.py`
- **Results**: `papers/emr-dynamic-functions-bio-inspired/results/tier2_replication/` (11 JSON files)
- **Runs**: 300 runs

**CL sequence**: XOR → Parity-3 → Parity-4 → Parity-5 (re-initialized NEAT, strategy persists)
**Key metric**: `sin_retained_after_discovery` (not `sin_retained_all_tasks`, XOR doesn't need sin)

#### Exp 3A: Memory Cell Phase Transition (7 strategies × 30 seeds = 210 runs)

| Strategy | MC Count | Retention % | 95% CI |
|---|---|---|---|
| Consolidation Window | 9.4 | 100% | [89, 100] |
| Clonal Selection | 24.0 | 100% | [89, 100] |
| Eligibility Trace | 10.0 | 80% | [63, 91] |
| Neuromodulated | 10.3 | 53.3% | [36, 70] |
| Metaplastic |— |— |— |
| Ecological Succession |— |— |— |
| Predictive Coding |— |— |— |

**Finding**: NO clean phase transition. Strategy type matters more than MC count. Pooled retention: 81.4% overall (171/210).

#### Exp 3B: Protected Index Ablation (3 strategies × 30 seeds × 2 conditions = 180 runs)

| Strategy | Protected Ret.% | Unprotected Ret.% | Δ |
|---|---|---|---|
| Clonal Selection | 100% | 100% | 0pp |
| Eligibility Trace | 80% | 80% | 0pp |
| Consolidation Window | 100% | 100% | 0pp |

**Finding**: Protection deltas are 0pp at N=30. Memory cells form independently within ~10 generations, so affinity floors are redundant at scale.

---

### Exp 3 Summary

| Workstream | Experiments | Strategies |
|---|---|---|
| 3A (MC Phase Transition) | 210 | 7 |
| 3B (Protection Ablation) | 90 | 3 × 2 |
| **Exp 3 Total** | **300** |— |

### Grand Total

| Workstream | Experiments |
|---|---|
| A-E (original) | 3,000 |
| Exp 3 (Tier 2 replication) | 300 |
| Persistent CL (Exp 1) | 420 |
| Oracle baseline (Exp 5) | 90 |
| **Grand Total** | **~3,810** |

Note: Some experiments overlap between workstreams (e.g., oracle baseline counted in Workstream totals). Paper reports 1,650 unique experimental runs.

---

## Statistical Methods

All analyses use non-parametric methods appropriate for N=30:

- **Solve rates**: Proportion ± 95% Wilson binomial CI
- **Convergence speed**: Median generations ± bootstrapped 95% CI (10,000 resamples)
- **2-group comparisons**: Mann-Whitney U + rank-biserial effect size
- **Multi-group comparisons**: Kruskal-Wallis H-test + post-hoc Dunn's test (Bonferroni)
- **CL metrics**: FWT, BWT, AA, Sin Retention as defined in benchmark_continual_learning.py
