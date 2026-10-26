# Bio-Inspired Palette Evolution: N=30 Benchmark Results

Benchmark results for the PPSN 2026 paper *"Bio-Inspired Palette Evolution in
Indirectly Encoded Substrates"*.

> **Note:** the `scripts/benchmarks/emr_optimization/...` script paths named below are from the
> original research monorepo and are not shipped in this standalone repository; the released
> runners live under `scripts/runners/`. Result data ships via the data release (root
> `scripts/fetch_results.py`).

## Benchmarks

### Benchmark A: Single-Task 8-Strategy Comparison (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/stdp_hebbian_replication.py`
- **Paper**: Table 1 (correction: Table 3 in the published paper)
- **Runs**: 240 experiments
- **Output dir**: `scripts/benchmarks/emr_optimization/results/stdp_hebbian_replication_20260206_163126/`
- **Data release**: `results/single_task/` (only the Predator-Prey row differs)

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

**Correction:** Predator-Prey solved 23 of 30 runs (77%, Wilson [0.591, 0.882], median 39) in
the paper's Table 3 and in `results/single_task/predator_prey.json`; the row above disagrees. On
those records the Kruskal-Wallis test above (generations to solve of the solved runs, 8
strategies) gives H=38.89, p=2.05e-6; H=54.16 comes from an earlier research run with the
Predator-Prey row printed above.

### Benchmark B: Continual Learning, Elite Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/run_cl_elite_sequential.sh`
- **Paper**: Table 2 (correction: the published paper has no continual-learning results)
- **Runs**: 1,470 experiments = 7 strategies x 30 seeds x 7 tasks
- **Output dir**: `results/continual_learning_elite_n30/`
- **Data release**: `results/cl_reinitialized/`

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

**Finding**: All strategies produce **identical** solve/fail patterns per seed.
Strategies affect palette composition (sin retention) but not evolutionary trajectory.
NEAT seed determinism overrides strategy effects on topology evolution.
(correction: this mechanism is not shown; AA is 0.835 for every strategy while sin retention
runs from 0% to 100%, so why the palette leaves the trajectory unchanged is open.)
FWT = +85.9% across all strategies (later parity tasks trivially solved after earlier ones).

### Benchmark C: Continual Learning, Hybrid Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/run_cl_hybrid_sequential.sh`
- **Paper**: FWT/BWT (correction: the published paper has no continual-learning results)
- **Runs**: 1,050 experiments = 5 strategies x 30 seeds x 7 tasks
- **Output dir**: `results/continual_learning_hybrid_n30/`
- **Data release**: `results/cl_hybrid/`

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

### Benchmark D: Dual-Domain Bio Strategies (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_bio_dual_palette.py`
- **Paper**: dual-domain discovery (correction: the paper's joint-system Parity-4 rates,
  Table 7, come from other runs)
- **Runs**: 150 experiments
- **Output dir**: `results/bio_dual_palette_20260206_163127/`
- **Data release**: `results/dual_domain/`

**Results**:

| Strategy | Solve% | Sin Disc.% | Sin Ret.% | Agg Disc.% |
|---|---|---|---|---|
| **Clonal Selection** | **90%** | 3% | 0% | 3% |
| Circadian | 87% | 93% | 50% | 0% |
| STDP | 77% | 67% | 37% | 50% |
| Baseline | 77% | 43% | 37% | 0% |
| Crit. Period Refined | 77% | 50% | 13% | 0% |

### Benchmark E: Failed Mechanisms (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_bio_dual_palette.py`
- **Paper**: failed mechanisms
- **Runs**: 90 experiments
- **Output dir**: `results/bio_dual_palette_20260206_163130/`
- **Data release**: `results/failure_mechanisms/`

**Results**:

| Mechanism | Solve Rate | 95% CI | Root Cause |
|---|---|---|---|
| GRN | 3% | [0.01, 0.17] | Expression dynamics too slow |
| Glial Modulation | 47% | [0.30, 0.64] | Modulatory timescale exceeds eval |
| Ant Colony Pheromone | 80% | [0.63, 0.90] | Pheromone accumulation lag |

**Correction:** the root causes above are the paper's timescale explanation. The rescaling
experiment's GRN arm, the paper's causal evidence, does not show causation: the rescaled GRN
ends at palette [16, 17] in 30 of 30 seeds, solved or not, so the root causes are proposed
causes.

---

## Summary Statistics

| Benchmark | Experiments | Strategies |
|---|---|---|
| A (Single-Task) | 240 | 8 |
| B (CL Elite) | 1,470 | 7 |
| C (CL Hybrid) | 1,050 | 5 |
| D (Dual-Domain) | 150 | 5 |
| E (Failed) | 90 | 3 |
| **Total (A-E)** | **3,000** | N/A |

---

### Exp 3: Tier 2 Replication (N=30)
- **Script**: `scripts/benchmarks/emr_optimization/benchmark_tier2_replication.py`
- **Analysis**: `scripts/benchmarks/emr_optimization/analysis_tier2_replication.py`
- **Results**: `papers/emr-dynamic-functions-bio-inspired/results/tier2_replication/` (11 JSON files)
- **Runs**: 300 runs

**CL sequence**: XOR -> Parity-3 -> Parity-4 -> Parity-5 (re-initialized NEAT, strategy persists)
**Key metric**: `sin_retained_after_discovery` (not `sin_retained_all_tasks`, XOR doesn't need sin)

#### Exp 3A: Memory Cell Phase Transition (7 strategies x 30 seeds = 210 runs)

| Strategy | MC Count | Retention % | 95% CI |
|---|---|---|---|
| Consolidation Window | 9.4 | 100% | [89, 100] |
| Clonal Selection | 24.0 | 100% | [89, 100] |
| Eligibility Trace | 10.0 | 80% | [63, 91] |
| Neuromodulated | 10.3 | 53.3% | [36, 70] |
| Metaplastic | not reported | not reported | not reported |
| Ecological Succession | not reported | not reported | not reported |
| Predictive Coding | not reported | not reported | not reported |

**Finding**: No clean phase transition. Strategy type matters more than MC count. Pooled retention: 81.4% overall (171/210).

**Correction:** `results/tier2_replication/analysis_summary.json` holds the three rows not
reported above, which with the four rows above give the pooled 171/210: Metaplastic
20.7 memory cells, 22/30 retained (73.3%, [56, 86]); Ecological Succession 17.3, 26/30 (86.7%,
[70, 95]); Predictive Coding 10.0, 23/30 (76.7%, [59, 88]).

#### Exp 3B: Protected Index Ablation (3 strategies x 30 seeds x 2 conditions = 180 runs)

| Strategy | Protected Ret.% | Unprotected Ret.% | Δ |
|---|---|---|---|
| Clonal Selection | 100% | 100% | 0pp |
| Eligibility Trace | 80% | 80% | 0pp |
| Consolidation Window | 100% | 100% | 0pp |

**Finding**: Protection deltas are 0pp at N=30. Memory cells form independently within ~10 generations, so affinity floors are redundant at scale. (The protected arm reuses the Exp 3A runs of these three strategies, so Exp 3B adds 90 runs.)

---

### Exp 3 Summary

| Part | Experiments | Strategies |
|---|---|---|
| 3A (MC Phase Transition) | 210 | 7 |
| 3B (Protection Ablation) | 90 | 3 x 2 |
| **Exp 3 Total** | **300** | N/A |

### Grand Total

| Benchmark | Experiments |
|---|---|
| A-E (original) | 3,000 |
| Exp 3 (Tier 2 replication) | 300 |
| Persistent CL (Exp 1) | 420 |
| Oracle baseline (Exp 5) | 90 |
| **Grand Total** | **~3,810** |

Note: Some experiments overlap between benchmarks (e.g., oracle baseline counted in benchmark totals). Paper reports 1,650 unique experimental runs.
(correction: the published paper reports more than 3,000 runs and does not state 1,650.)

---

## Statistical Methods

All analyses use non-parametric methods appropriate for N=30:

- **Solve rates**: Proportion ± 95% Wilson binomial CI
- **Convergence speed**: Median generations ± bootstrapped 95% CI (10,000 resamples)
- **2-group comparisons**: Mann-Whitney U + rank-biserial effect size
- **Multi-group comparisons**: Kruskal-Wallis H-test + post-hoc Dunn's test (Bonferroni)
- **CL metrics**: FWT, BWT, AA, Sin Retention as defined in benchmark_continual_learning.py
