# ALIFE 2026 Neuromodulation Paper: Experiment Results

**Paper**: "Multi-Behavioral Evolved Substrates Through Neuromodulation and Activation Selection"

---

## Summary of All Experiments

| Dataset | N | Schema | Key Config | Purpose |
|---------|---|--------|-----------|---------|
| Benchmark | 1,665 | A | Uniform tanh, varied topology/pop | Map multi-task feasibility |
| Ablation | 744 | A | Varied architecture, Pop=750, 200 gen | 2-task compatibility (χ² stats) |
| Validation (5-task) | 30 | B | Per-task activation, Pop=750, product | Validate 100% success |
| Validation (2-task) | 300 | B | Per-task activation, Pop=750, 30 seeds/pair | Validate pair compatibility |
| Component Ablation | 90 | B | 3 conditions × 30 seeds | Isolate critical components |
| Schema A uniform n=30 | 300 | A | Uniform tanh, Pop=750, 30 seeds/pair | Table 2 column 1 |
| Schema B uniform n=30 | 300 | B | Uniform tanh, Pop=750, 30 seeds/pair | Table 2 column 2 |
| Topology Validation | 360 | B | 6 topologies × 30 seeds × 2 agg | Verify recurrence neutrality |
| Uniform Sin Baseline | 30 | B | All sin, Pop=750, 30 seeds | Test if sin alone suffices |
| Neuromod Ablation | 30 | B | Flat NT, per-task activation, 30 seeds | Test if neuromod is necessary |
| Random NT (Exp 13) | 600 | B | 10 random NT sets × 30 seeds × 2 inv | W4: hand-design criticism |
| Task Scaling (Exp 14) | 120 | B | 6/7/8/10 tasks × 30 seeds | W3: toy 5-task scale |
| Sigma Sweep (Exp 15) | 180 | B | 6 strengths × 30 seeds | W: unjustified s=5.0 |
| Deeper Substrate (Exp 16) | 120 | B | 3 depths uniform + d=6 control × 30 seeds | W2: single-layer |
| Gain/Bias Isolation | 30 | B | Flat NT + ACh polarity, 30 seeds | Isolate gain/bias from inversion |
| Continuous Domain (Exp 18) | 90 |— | 3 tasks × 3 conditions × 30 seeds | Domain specificity test |
| Multi-Head Control (E17) | 120 |— | 2 arch × 2 opt × 30 seeds | Architecture specificity |
| Multi-Layer Neuromod (E18) | 300 | B | 5 depths × 2 act × 30 seeds | Multi-layer depth test |
| Higher-Dim NT (E19) | 120 | B+ | 4 cond × 30 seeds | NT dimensionality test |
| Multi-Layer 7-Task (E20) | 120 | B | 4 cond × 30 seeds | 7-task depth test |
| IMPLY Barrier (E21) | 180 | B | 6 cond × 30 seeds | Activation-level mechanism |
| Higher Arity (E22) | 120 | B | 4 cond × 30 seeds | 4-input barrier test |
| Higher Arity Capacity (E22b) | 120 | B | 4 cond × 30 seeds | Capacity control |
| Frozen Substrate (E24) | 120 | B | 4 cond × 30 seeds | Training vs eval barrier |
| Higher Arity 5-Input (E25) | 120 | B | 4 cond × 30 seeds | 5-input barrier test |
| Indirect Encoding Depth (E26) | 90 | B | 3 cond × 30 seeds | CPPN + depth |
| Task Scaling 2-Layer (E27) | 120 | B | 4 cond × 30 seeds | 8/10-task scaling |
| 2D Synthetic Classif. (E28) | 120 |— | 4 cond × 30 seeds | Domain generality |
| **Total** | **6,639** | | | |

---

## 1. Validation: 30-Seed 5-Task Success

**Config**: Pop=750, 100 generations, product aggregation, per-task activation (sin for XOR, tanh for others), Schema B NT vectors, feedforward topology, ≥98% threshold.

**Result**: 30/30 seeds converge (100%)

### Convergence Statistics
- **Mean**: 15.9 generations
- **SD**: 8.8
- **Median**: 14.0
- **Range**: 4–36
- **95% CI**: [12.7, 19.2] (t-distribution, df=29)
- **IQR**: Q1=9, Q3=23

### Statistical Tests
- **Rule of Three**: 95% upper bound on failure rate = 10% → true success rate ≥ 90%
- **Exact binomial**: P(30/30 | true_rate=0.90) = 0.042 (reject H₀: rate ≤ 0.90)
- **Effect size**: Cohen's h ≈ 3.14 (theoretical maximum, comparing 0/54 baseline with 30/30)

### Per-Seed Data

#### Per-seed data (30 seeds)
| Seed | Convergence Gen | Runtime (s) |
|------|----------------|-------------|
| 42 | 11 | 39 |
| 123 | 7 | 24 |
| 456 | 31 | 83 |
| 789 | 6 | 26 |
| 1000 | 6 | 24 |
| 1 | 32 | 86 |
| 2 | 9 | 30 |
| 3 | 7 | 27 |
| 4 | 20 | 59 |
| 5 | 13 | 48 |
| 6 | 14 | 66 |
| 7 | 10 | 49 |
| 8 | 23 | 91 |
| 9 | 15 | 62 |
| 10 | 16 | 61 |
| 11 | 36 | 123 |
| 12 | 14 | 58 |
| 13 | 24 | 87 |
| 14 | 31 | 118 |
| 15 | 13 | 55 |
| 16 | 26 | 100 |
| 17 | 11 | 51 |
| 18 | 16 | 65 |
| 19 | 23 | 90 |
| 20 | 9 | 45 |
| 21 | 11 | 53 |
| 22 | 15 | 64 |
| 23 | 6 | 35 |
| 24 | 19 | 78 |
| 25 | 4 | 30 |

#### All 30 Seeds (sorted by convergence gen)
```
[4, 6, 6, 6, 7, 7, 9, 9, 10, 11, 11, 11, 13, 13, 14, 14, 15, 15, 16, 16, 19, 20, 23, 23, 24, 26, 31, 31, 32, 36]
```

All 30 seeds: 100% fitness on all 5 tasks (XOR, AND, OR, NAND, NOR). Every seed uses: sin activation for XOR, tanh for all threshold tasks. Per-task fitness = 1.0 for all tasks in all seeds.

---

## 2. Component Ablation

**Config**: Pop=750, 200 generations, ≥98% threshold, 30 seeds per condition, Schema B.

### Condition A: Full Baseline (per-task activation + neuromodulation + product aggregation)
- **Result**: 30/30 converged (100%)
- **Median**: 14 generations | **Mean**: 15.9 | **SD**: 8.8
- **95% CI**: [12.7, 19.2] | **Range**: 4–36

#### Per-seed data (30 seeds)
| Seed | Gen |
|------|-----|
| 42 | 11 |
| 123 | 7 |
| 456 | 31 |
| 789 | 6 |
| 1000 | 6 |
| 1 | 32 |
| 2 | 9 |
| 3 | 7 |
| 4 | 20 |
| 5 | 13 |
| 6 | 14 |
| 7 | 10 |
| 8 | 23 |
| 9 | 15 |
| 10 | 16 |
| 11 | 36 |
| 12 | 14 |
| 13 | 24 |
| 14 | 31 |
| 15 | 13 |
| 16 | 26 |
| 17 | 11 |
| 18 | 16 |
| 19 | 23 |
| 20 | 9 |
| 21 | 11 |
| 22 | 15 |
| 23 | 6 |
| 24 | 19 |
| 25 | 4 |

#### All 30 sorted
```
[4, 6, 6, 6, 7, 7, 9, 9, 10, 11, 11, 11, 13, 13, 14, 14, 15, 15, 16, 16, 19, 20, 23, 23, 24, 26, 31, 31, 32, 36]
```

### Condition B: No Per-Task Activation (uniform tanh)
- **Result**: 0/30 converged (0%)
- **All 30 seeds**: XOR stuck at 75%, AND/OR/NAND/NOR at 100%
- **Min fitness**: 0.75 for all seeds (the XOR ceiling)

### Condition C: No Product Aggregation (min instead of product)
- **Result**: 30/30 converged (100%)
- **Median**: 17 generations | **Mean**: 19.9 | **SD**: 15.5
- **95% CI**: [14.1, 25.7] | **Range**: 3–78

#### Per-seed data (30 seeds)
| Seed | Gen |
|------|-----|
| 42 | 19 |
| 123 | 17 |
| 456 | 25 |
| 789 | 78 |
| 1000 | 18 |
| 1 | 40 |
| 2 | 10 |
| 3 | 22 |
| 4 | 15 |
| 5 | 23 |
| 6 | 5 |
| 7 | 15 |
| 8 | 4 |
| 9 | 16 |
| 10 | 8 |
| 11 | 28 |
| 12 | 15 |
| 13 | 10 |
| 14 | 6 |
| 15 | 7 |
| 16 | 12 |
| 17 | 53 |
| 18 | 21 |
| 19 | 12 |
| 20 | 26 |
| 21 | 3 |
| 22 | 23 |
| 23 | 13 |
| 24 | 35 |
| 25 | 18 |

#### All 30 sorted
```
[3, 4, 5, 6, 7, 8, 10, 10, 12, 12, 13, 15, 15, 15, 16, 17, 18, 18, 19, 21, 22, 23, 23, 25, 26, 28, 35, 40, 53, 78]
```

### Summary

| Condition | Rate | Median Gen | Mean Gen | SD | 95% CI | Range |
|-----------|------|-----------|---------|-----|--------|-------|
| full_baseline | 30/30 (100%) | 14 | 15.9 | 8.8 | [12.7, 19.2] | 4–36 |
| no_pertask_activation | 0/30 (0%) |— |— |— |— | all 0.75 |
| no_product_aggregation | 30/30 (100%) | 17 | 19.9 | 15.5 | [14.1, 25.7] | 3–78 |

### Key Finding
Product aggregation is NOT required for success. It accelerates convergence by 1.2× (median 17 vs 14). Per-task activation IS required (0/30 without it). The old 5-seed estimate of 2.7× slowdown (median 19 vs 7) was inflated by small sample size.

---

## 3. 2-Task Ablation with Schema B (Per-Task Activation)

**Config**: Pop=750, 200 generations, ≥98% threshold, per-task activation, product aggregation, Schema B NT vectors, 30 seeds per pair.

### Summary: ALL 10 Pairs → 100%

| Pair | Convergence | Perfection | Schema A Rate (uniform tanh) |
|------|-------------|------------|------------------------------|
| XOR+AND | 30/30 (100%) | 30/30 (100%) | 12.5% |
| XOR+OR | 30/30 (100%) | 30/30 (100%) | 79.2% |
| XOR+NAND | 30/30 (100%) | 30/30 (100%) | 70.8% |
| XOR+NOR | 30/30 (100%) | 30/30 (100%) | 83.3% |
| AND+OR | 30/30 (100%) | 30/30 (100%) | 100% |
| AND+NAND | 30/30 (100%) | 30/30 (100%) | 100% |
| AND+NOR | 30/30 (100%) | 30/30 (100%) | 33.3% |
| OR+NAND | 30/30 (100%) | 30/30 (100%) | 100% |
| OR+NOR | 30/30 (100%) | 30/30 (100%) | 41.7% |
| NAND+NOR | 30/30 (100%) | 30/30 (100%) | 100% |

### Key Finding
Per-task activation **completely eliminates** all compatibility barriers. The 2-task compatibility variation under uniform activation (Table 5 in paper) is entirely attributable to activation function mismatch. XOR+AND goes from 12.5% → 100%; OR+NOR goes from 41.7% → 100%.

---

## 4. NT Vector Schemas

### Schema A (NT_PRESETS_4): Used in Benchmark (1,665) and Ablation (744)

| Task | DA | 5HT | NE | ACh |
|------|-----|------|-----|------|
| XOR | 0.95 | 0.05 | 0.95 | 1.0 |
| AND | 0.10 | 0.90 | 0.10 | 1.0 |
| OR | 0.50 | 0.50 | 0.50 | 1.0 |
| NAND | 0.90 | 0.10 | 0.50 | 1.0 |
| NOR | 0.10 | 0.50 | 0.90 | 1.0 |

Each task has a unique NT vector. ACh=1.0 for all (no output inversion).

### Schema B (HEAD_NT_PROFILES): Used in Validation (30 seeds, 300 2-task, 90 component)

| Task | DA | 5HT | NE | ACh |
|------|-----|------|-----|------|
| XOR | 0.95 | 0.05 | 0.95 | 1.0 |
| AND | 0.10 | 0.90 | 0.10 | 1.0 |
| OR | 0.50 | 0.50 | 0.50 | 1.0 |
| NAND | 0.10 | 0.90 | 0.10 | 0.0 |
| NOR | 0.50 | 0.50 | 0.50 | 0.0 |

NAND = AND + ACh inversion. NOR = OR + ACh inversion. Three unique modulatory profiles + two inversions.

### Key Differences
- XOR, AND, OR: **Identical** across both schemas
- NAND: Schema A [0.90, 0.10, 0.50, 1.0] vs Schema B [0.10, 0.90, 0.10, 0.0]
- NOR: Schema A [0.10, 0.50, 0.90, 1.0] vs Schema B [0.50, 0.50, 0.50, 0.0]

---

## 5. Activation Functions

| Task | Activation | Rationale |
|------|-----------|-----------|
| XOR | sin | Oscillatory, matches parity's alternating output pattern |
| AND | tanh | Bounded monotonic, matches threshold detection |
| OR | tanh | Bounded monotonic, matches threshold detection |
| NAND | tanh | Bounded monotonic, matches threshold detection (+ ACh inversion in Schema B) |
| NOR | tanh | Bounded monotonic, matches threshold detection (+ ACh inversion in Schema B) |

**Note**: The code defines `HEAD_ACTIVATIONS = {'xor': 'sin', 'and': 'tanh', 'or': 'sigmoid', ...}` but the experiment runner function (`run_multihead_palette_experiment`) uses `'sin' if t == 'xor' else 'tanh'`, so OR actually uses tanh, not sigmoid. All JSON results confirm `"or": "tanh"`.

---

## 6. Data File Locations

| Dataset | Location |
|---------|----------|
| Validation 25 new seeds | `results/validation_30seeds/seed_*.json` |
| Validation 5 original seeds | `results/palette_per_head_20260118_185*.json` |
| Component ablation | `results/component_ablation/component_ablation_20260206_015511.json` |
| 2-task Schema B per-task (per pair) | `results/ablation_schema_b/pair_*.json` |
| 2-task Schema B per-task (summary) | `results/ablation_schema_b/ablation_schema_b_20260206_020603.json` |
| 2-task Schema A uniform n=30 | `papers/emr-neuromodulation/results/ablation_schema_a_uniform/pair_*.json` |
| 2-task Schema B uniform n=30 | `papers/emr-neuromodulation/results/ablation_schema_b_uniform/pair_*.json` |
| Benchmark 1,665 | See `docs/research/neuromodulation_statistical_analysis.md` |
| Ablation 744 | See `docs/research/neuromodulation_ablation_study.md` |

---

## 7. Experiment Scripts

> **Note:** the `experiments/neuromodulation/...` script paths and `docs/research/...` documents
> named in this file are from the original research monorepo and are not shipped in this
> standalone repository. The released runners live under `scripts/runners/` (this paper's
> [README](README.md) lists them); result data ships via the data release (root
> `scripts/fetch_results.py`).

| Script | Purpose |
|--------|---------|
| `experiments/neuromodulation/run_validation_30seeds.py` | 25 additional 5-task validation seeds |
| `experiments/neuromodulation/run_2task_ablation_schema_b.py` | All 10 pairs × seeds 1-24 with Schema B |
| `experiments/neuromodulation/run_2task_extension_seeds25_30.py` | Extend 2-task from 24 to 30 seeds |
| `experiments/neuromodulation/run_component_ablation.py` | 3 conditions × 5 seeds (original) |
| `experiments/neuromodulation/run_component_ablation_extended.py` | 3 conditions × 25 seeds (extended to 30) |
| `experiments/neuromodulation/run_2task_schema_a_uniform.py` | Schema A + uniform tanh, 10 pairs × 30 seeds |
| `experiments/neuromodulation/run_2task_schema_b_uniform.py` | Schema B + uniform tanh, 10 pairs × 3 seeds (original) |
| `experiments/neuromodulation/run_2task_schema_b_uniform_extension.py` | Extend Schema B uniform from 3 to 30 seeds |
| `experiments/neuromodulation/multihead_palette_neuromodulation.py` | Core experiment runner |

---

## 8. Cross-Schema Comparison: Uniform Tanh 2-Task (n=30)

**Config**: Pop=750, 100 generations, ≥98% threshold, feedforward, product aggregation, UNIFORM tanh (palette_mode='uniform'). 10 pairs × 30 seeds = 300 experiments per schema.

**Purpose**: Isolate activation function variable from NT schema variable with full statistical power. Both schemas now have n=30 per pair, matching the per-task activation condition.

### Results (n=30 per pair)

| Pair | Schema A (Uniform) | Schema B (Uniform) | Schema B (Per-task) |
|------|-------------------|-------------------|---------------------|
| AND+OR | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| AND+NAND | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| AND+NOR | 16.7% (5/30) | **100%** (30/30) | 100% (30/30) |
| OR+NAND | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| OR+NOR | **0%** (0/30) | **100%** (30/30) | 100% (30/30) |
| NAND+NOR | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| XOR+OR | 86.7% (26/30) | 86.7% (26/30) | 100% (30/30) |
| XOR+NOR | **100%** (30/30) | 83.3% (25/30) | 100% (30/30) |
| XOR+AND | 0% (0/30) | 0% (0/30) | 100% (30/30) |
| XOR+NAND | 46.7% (14/30) | **0%** (0/30) | 100% (30/30) |

### Key Findings (strengthened at n=30)

1. **Schema B NAND = AND at hidden layer**: Under Schema B, NAND has DA/5HT/NE=[0.10, 0.90, 0.10] (same as AND). Only ACh differs. Since ACh doesn't affect hidden layer modulation, XOR+NAND under Schema B behaves like XOR+AND → both 0%.

2. **Schema B NOR = OR at hidden layer**: NOR has DA/5HT/NE=[0.50, 0.50, 0.50] (same as OR). So:
   - XOR+NOR ≈ XOR+OR (83.3% vs 86.7%, near-identical)
   - OR+NOR = OR+OR effectively → 100% (no modulation conflict)
   - AND+NOR = AND+OR effectively → 100%

3. **Activation function is the sole barrier**: XOR-containing pairs fail regardless of NT schema. Threshold-only pairs succeed in both schemas (with differences explained by effective modulation equivalences).

4. **New at n=30**: Schema A reveals OR+NOR=0% and XOR+NOR=100% (the opposite of Schema B), confirming that NT vector geometry, not just activation, determines inter-schema pair rearrangement. AND+NOR drops from the old 33.3% (n=24 mixed-config) to 16.7% (n=30 single-config), suggesting the mixed-config ablation overestimated some rates.

### Data Locations

| Dataset | Script | Results |
|---------|--------|---------|
| Schema A uniform n=30 | `run_2task_schema_a_uniform.py` | `papers/emr-neuromodulation/results/ablation_schema_a_uniform/` |
| Schema B uniform n=30 | `run_2task_schema_b_uniform.py` + `run_2task_schema_b_uniform_extension.py` | `papers/emr-neuromodulation/results/ablation_schema_b_uniform/` |

---

## 9. Previously Verified Claims (from ablation study)

| Claim | Value | Source |
|-------|-------|--------|
| Total benchmark experiments | 1,665 | neuromodulation_statistical_analysis.md |
| Total ablation experiments | 744 | neuromodulation_ablation_study.md |
| χ² for task count | 382.71 | neuromodulation_ablation_study.md |
| Cramér's V | 0.717 | neuromodulation_ablation_study.md |
| Logistic β (task count) | -1.521 | scripts/analysis/neuromodulation_statistical_analysis.py:758-822 |
| XOR+AND convergence (Schema A, mixed-config n=24) | 12.5% (3/24) | neuromodulation_ablation_study.md |
| XOR+AND convergence (Schema A, single-config n=30) | 0% (0/30) | ablation_schema_a_uniform/pair_xor+and.json |
| XOR+OR convergence (Schema A, mixed-config n=24) | 79.2% (19/24) | neuromodulation_ablation_study.md |
| XOR+OR convergence (Schema A, single-config n=30) | 86.7% (26/30) | ablation_schema_a_uniform/pair_xor+or.json |
| AND+NAND convergence (Schema A) | 100% (24/24 and 30/30) | neuromodulation_ablation_study.md |
| AND+NAND+OR 3-task (Schema A) | 79.2% | neuromodulation_ablation_study.md |

---

## 10. Topology Validation (B1: Product + B2: Min)

**Config**: Pop=750, 200 generations, ≥98% threshold, per-task activation (sin for XOR, tanh for others), Schema B NT vectors, 30 seeds (1000–1029) per topology.

**Six topologies tested**: feedforward, hidden_only, with_backward, with_lateral, with_self, full_recurrent.

### B1: Product Aggregation (180/180 converged)

| Stat | Value |
|------|-------|
| Converged | 180/180 (100%) |
| Median | 9.5 generations |
| Mean | 12.7 |
| SD | 12.5 |
| 95% CI (bootstrap, median) | [5, 13] |
| Range | 0–51 |

### B2: Min Aggregation (180/180 converged)

| Stat | Value |
|------|-------|
| Converged | 180/180 (100%) |
| Median | 18.0 generations |
| Mean | 21.0 |
| SD | 13.6 |
| 95% CI (bootstrap, median) | [14.5, 23.0] |
| Range | 6–55 |

### Key Finding

**Recurrence is strictly neutral under both aggregation methods.** All six topologies produce **identical** convergence generation per seed: the same seed converges at the same generation regardless of recurrence configuration. This holds for both product and min aggregation.

Min aggregation is ~1.9× slower than product (median 18 vs 9.5), consistent with the Component Ablation result (median 17 vs 14 for the 30-seed non-topology experiment).

### Per-Seed Convergence (all topologies identical)

| Seed | B1 (product) | B2 (min) |
|------|-------------|----------|
| 1000 | 5 | 18 |
| 1001 | 22 | 27 |
| 1002 | 24 | 16 |
| 1003 | 13 | 34 |
| 1004 | 32 | 18 |
| 1005 | 9 | 19 |
| 1006 | 33 | 8 |
| 1007 | 21 | 23 |
| 1008 | 4 | 8 |
| 1009 | 31 | 7 |
| 1010 | 13 | 13 |
| 1011 | 3 | 7 |
| 1012 | 5 | 21 |
| 1013 | 2 | 23 |
| 1014 | 15 | 19 |
| 1015 | 0 | 54 |
| 1016 | 4 | 6 |
| 1017 | 3 | 28 |
| 1018 | 5 | 44 |
| 1019 | 31 | 6 |
| 1020 | 14 | 26 |
| 1021 | 12 | 55 |
| 1022 | 8 | 18 |
| 1023 | 4 | 9 |
| 1024 | 3 | 9 |
| 1025 | 51 | 24 |
| 1026 | 10 | 44 |
| 1027 | 1 | 17 |
| 1028 | 3 | 10 |
| 1029 | 19 | 18 |

### Sorted Convergence Generations

**B1 (product)**: [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 8, 9, 10, 12, 13, 13, 14, 15, 19, 21, 22, 24, 31, 31, 32, 33, 51]

**B2 (min)**: [6, 6, 7, 7, 8, 8, 9, 9, 10, 13, 16, 17, 18, 18, 18, 18, 19, 19, 21, 23, 23, 24, 26, 27, 28, 34, 44, 44, 54, 55]

### Data Files

| Dataset | Location |
|---------|----------|
| B1 product (6 topologies) | `results/topology_validation/5task_product_*.json` |
| B2 min (6 topologies) | `papers/emr-neuromodulation/results/topology_validation/5task_min_*.json` |

**Note**: B2 feedforward has 3 files (first two from pre-fix runs with 11/30 and 13/30 convergence). The canonical file is `5task_min_feedforward_20260209_125607.json` (30/30, post-fix).

**Script**: `experiments/neuromodulation/run_topology_validation.py`
**Total runtime**: B1 ~11,066s (full_recurrent) to ~2,088s (feedforward); B2 similar range.

---

## 11. Uniform Sin Baseline (Experiment A)

**Config**: Pop=750, 200 generations, ≥98% threshold, ALL tasks use sin activation (uniform_sin), Schema B NT vectors, product aggregation, 30 seeds (1-30).

**Purpose**: Test if sin activation alone suffices for all 5 tasks, or if diversity (sin for XOR, tanh for threshold) is structurally necessary.

### Result: 30/30 converged (100%)

| Stat | Value |
|------|-------|
| Converged | 30/30 (100%) |
| Median | 17 generations |
| Range | 2–32 |

### Per-Seed Data
| Seed | Gen | Seed | Gen | Seed | Gen |
|------|-----|------|-----|------|-----|
| 1 | 18 | 11 | 12 | 21 | 29 |
| 2 | 17 | 12 | 18 | 22 | 17 |
| 3 | 11 | 13 | 29 | 23 | 17 |
| 4 | 12 | 14 | 32 | 24 | 9 |
| 5 | 18 | 15 | 14 | 25 | 11 |
| 6 | 5 | 16 | 20 | 26 | 16 |
| 7 | 17 | 17 | 2 | 27 | 16 |
| 8 | 30 | 18 | 14 | 28 | 15 |
| 9 | 17 | 19 | 27 | 29 | 12 |
| 10 | 6 | 20 | 17 | 30 | 23 |

### Key Finding

**Sin alone works for ALL 5 tasks.** The barrier is specifically *monotonic* activation (tanh), not lack of diversity. Sin can approximate threshold functions via its monotonically increasing region, while also handling parity natively. However, per-task assignment provides faster convergence: median 14 (per-task) vs 17 (uniform sin).

### Data Location
`papers/emr-neuromodulation/results/uniform_sin_baseline/`

**Script**: `experiments/neuromodulation/run_uniform_sin_baseline.py`

---

## 12. Neuromodulation Ablation (Experiment B)

**Config**: Pop=750, 200 generations, ≥98% threshold, per-task activation (sin for XOR, tanh for threshold), FLAT NT vectors (identical [0.5, 0.5, 0.5, 1.0] for all tasks, ACh=1.0), product aggregation, 30 seeds (1-30).

**Purpose**: Test if neuromodulation is structurally necessary or merely an accelerator. Per-task activation is retained, but all task-specific modulation is removed.

### Result: 0/30 converged (0%), COMPLETE (all 30 seeds)

All 30 seeds plateau at min_fitness = 0.500.

**Diagnostic pattern** (100% consistent across all 30 seeds):
- XOR: 100% (30/30 seeds), sin activation succeeds without modulation
- AND: 50% (all 30 seeds), chance level
- OR: 50% (all 30 seeds), chance level
- NAND: 50% (all 30 seeds), chance level
- NOR: 50% (all 30 seeds), chance level

### Key Finding

**Neuromodulation IS structurally necessary.** Without task-specific modulation:
- XOR succeeds (sin activation suffices without modulation for parity)
- All four threshold tasks fail at exactly 50% (chance level for 2/4 patterns)
- The network settles on a single decision threshold that gets 2/4 patterns correct for each threshold task
- NAND/NOR have no output inversion mechanism (ACh=1.0 for all)
- The same weights must simultaneously produce 5 different Boolean functions with identical pre-activation modulation, structurally impossible for threshold differentiation

### Data Location
`papers/emr-neuromodulation/results/neuromod_ablation/`

**Script**: `experiments/neuromodulation/run_neuromod_ablation.py`

---

## 13. Random NT Vectors (Experiment 13: Strengthening W4)

**Config**: Pop=750, 100 generations, ≥98% threshold, per-task activation, product aggregation, Schema B structure, 30 seeds per condition.
**Purpose**: Test whether hand-designed NT vectors are critical or arbitrary. 10 random NT vector sets × 30 seeds × 2 inversion conditions = 600 runs.

### Setup
- **No inversion** (300 runs): Random DA/5HT/NE values, ACh=1.0 for all 5 tasks
- **With inversion** (300 runs): Same random DA/5HT/NE, but ACh=0.0 for NAND/NOR (preserving inversion mechanism)
- Random NT vectors drawn from Uniform(0,1) per dimension, seed=12345

### Results

| Condition | Converged | Rate |
|-----------|-----------|------|
| No inversion (all ACh=1.0) | 0/300 | 0.0% |
| With inversion (NAND/NOR ACh=0.0) | 40/300 | 13.3% |

### Per-Set Breakdown (With Inversion)

| Set | Rate | Median Gen | Notes |
|-----|------|-----------|-------|
| 0 | 0/30 (0.0%) |— | |
| 1 | 1/30 (3.3%) | 62 | |
| 2 | 0/30 (0.0%) |— | |
| 3 | 30/30 (100.0%) | 10 | Best random set |
| 4 | 7/30 (23.3%) | 76 | |
| 5 | 0/30 (0.0%) |— | |
| 6 | 0/30 (0.0%) |— | |
| 7 | 2/30 (6.7%) | 62 | |
| 8 | 0/30 (0.0%) |— | |
| 9 | 0/30 (0.0%) |— | |

### Key Findings

1. **ACh polarity is the critical design decision**: 0/300 without inversion vs 40/300 with inversion. Without output inversion, NAND/NOR are structurally unsolvable since they require negating the AND/OR response.
2. **NT vector geometry is NOT arbitrary**: Only 4/10 random sets achieve any success, and only 1/10 achieves 100%. The hand-designed NT vectors succeed because they satisfy specific modulatory geometry constraints, not by luck.
3. **One-in-ten match**: The best random set (set 3) achieves 100% with median 10 gen, comparable to hand-designed (median 14 gen). This shows the geometry constraints are satisfiable but specific.

### Data Location
`papers/emr-neuromodulation/results/strengthening/random_nt/`

**Script**: `experiments/neuromodulation/run_strengthening_experiments.py --experiment random_nt`

---

## 14. Task Scaling (Experiment 14: Strengthening W3)

**Config**: Pop=750, 100 generations, ≥98% threshold, per-task activation, product aggregation, extended NT profiles (Schema B style with ACh inversion for negated tasks), 30 seeds per task count.
**Purpose**: Test scaling beyond 5 tasks. 4 task counts × 30 seeds = 120 runs.

### Extended Tasks
- 6 tasks: XOR, AND, OR, NAND, NOR, XNOR
- 7 tasks: + IMPLY
- 8 tasks: + NIMPLY
- 10 tasks: + CONVERSE_IMPLY, CONVERSE_NIMPLY

### Results

| Tasks | Converged | Rate | Median Gen | Mean Gen | SD | Range | Avg min_f |
|-------|-----------|------|-----------|---------|-----|-------|-----------|
| 5 (baseline) | 30/30 | 100% | 14 | 15.9 | 8.8 | 4–36 | 1.000 |
| 6 (XNOR) | 30/30 | 100% | 16 | 18.2 | 13.2 | 3–63 | 1.000 |
| 7 | 0/30 | 0% |— |— |— |— | 0.500 |
| 8 | 0/30 | 0% |— |— |— |— | 0.658 |
| 10 | 0/30 | 0% |— |— |— |— | 0.750 |

### 6-Task XNOR Per-Seed Data

| Seed | Gen | Seed | Gen | Seed | Gen |
|------|-----|------|-----|------|-----|
| 0 | 16 | 10 | 12 | 20 | 22 |
| 1 | 3 | 11 | 14 | 21 | 19 |
| 2 | 4 | 12 | 6 | 22 | 41 |
| 3 | 9 | 13 | 17 | 23 | 17 |
| 4 | 22 | 14 | 17 | 24 | 37 |
| 5 | 11 | 15 | 14 | 25 | 18 |
| 6 | 3 | 16 | 17 | 26 | 29 |
| 7 | 15 | 17 | 6 | 27 | 43 |
| 8 | 5 | 18 | 63 | 28 | 19 |
| 9 | 12 | 19 | 20 | 29 | 16 |

Sorted: [3, 3, 4, 5, 6, 6, 9, 11, 12, 12, 14, 14, 15, 16, 16, 17, 17, 17, 17, 18, 19, 19, 20, 22, 22, 29, 37, 41, 43, 63]

### Key Findings

1. **6-task succeeds trivially**: XNOR shares XOR's hidden-layer modulation (DA=0.95, 5HT=0.05, NE=0.95) with ACh=0.0 for output inversion. Compositionally related tasks add zero modulatory burden (median 16 gen, comparable to 5-task median 14).
2. **Sharp cliff at 6→7 tasks**: 100% → 0%. Not a gradual degradation.
3. **Honest limitation**: The current architecture (single-layer, fixed population) cannot scale beyond 6 Boolean tasks with per-task activation + neuromodulation.
4. **Min_f diagnostic**: 7-task avg min_f=0.500 suggests some tasks at chance level, indicating the modulation space is saturated: 4D NT vectors cannot create 7 sufficiently distinct modulatory states.

### Data Location
`papers/emr-neuromodulation/results/strengthening/task_scaling/`

**Script**: `experiments/neuromodulation/run_strengthening_experiments.py --experiment task_scaling`

---

## 15. Modulation Strength Sweep (Experiment 15: Strengthening W)

**Config**: Pop=750, 100 generations, ≥98% threshold, per-task activation, product aggregation, Schema B NT vectors, 30 seeds per strength value.
**Purpose**: Test robustness to modulation strength s (default s=5.0). 6 strengths × 30 seeds = 180 runs.

### Results

| s | Converged | Rate | Median Gen | 95% CI | Range |
|---|-----------|------|-----------|--------|-------|
| 0.5 | 28/30 | 93.3% | 20 | [2.7, 97.0] | [2, 99] |
| 1.0 | 30/30 | 100% | 9 | [3.0, 39.3] | [3, 48] |
| 2.0 | 30/30 | 100% | 7 | [3.0, 15.8] | [3, 18] |
| 5.0 | 30/30 | 100% | 14 | [5.5, 33.1] | [4, 36] |
| 10.0 | 30/30 | 100% | 6 | [1.0, 14.7] | [1, 22] |
| 20.0 | 30/30 | 100% | 6 | [2.7, 12.0] | [2, 12] |

### Key Findings

1. **Robust across 40× range**: s=1.0 through s=20.0 all achieve 100% (120/120). Only s=0.5 shows slight degradation (93.3%).
2. **s=5.0 is not special**: Multiple values achieve equal or better convergence speed. s=2.0 is actually fastest (median 7 gen).
3. **Low modulation weakens**: s=0.5 produces 2 failures (93.3%): insufficient modulation strength means NT vectors produce too-subtle gain/bias differences to separate tasks.
4. **No upper ceiling observed**: Even s=20.0 works perfectly. Strong modulation doesn't cause instability.

### Data Location
`papers/emr-neuromodulation/results/strengthening/sigma_sweep/`

**Script**: `experiments/neuromodulation/run_strengthening_experiments.py --experiment sigma_sweep`

---

## 16. Deeper Substrate (Experiment 16: Strengthening W2)

**Config**: Pop=750, 100 generations, ≥98% threshold, product aggregation, Schema B NT vectors, 30 seeds per condition.
**Purpose**: Test if deeper substrates (max_depth 4/5/6 vs default 3) can overcome the tanh barrier without per-task activation. 120 runs total (90 uniform + 30 per-head control). Runtime: 13.6h.

### Results

| Condition | Converged | Rate | Avg min_f |
|-----------|-----------|------|-----------|
| depth=4 uniform tanh | 0/30 | 0% | 0.750 |
| depth=5 uniform tanh | 0/30 | 0% | 0.750 |
| depth=6 uniform tanh | 0/30 | 0% | 0.750 |
| depth=6 per-head (control) | 30/30 | 100% | 1.000 |

### Key Findings

1. **Barrier is mathematical, not architectural**: Increasing substrate depth from 3→4→5→6 has zero effect. All uniform tanh conditions plateau at exactly 0.750 (the XOR ceiling).
2. **Deeper substrates cannot overcome monotonic activation limits**: A composition of monotonic functions is still monotonic. No depth of tanh layers can produce the oscillatory behavior XOR requires.
3. **Control confirmed**: depth=6 with per-task activation achieves 30/30 (100%, median 14 gen, range [4, 36]), identical to default-depth results. The barrier is activation-specific, not depth-related.

### Data Location
`papers/emr-neuromodulation/results/strengthening/deeper_substrate/`

**Script**: `experiments/neuromodulation/run_strengthening_experiments.py --experiment deeper_substrate`

---

## 17. Gain/Bias Isolation: Honest Neuromodulation Ablation

**Config**: Pop=750, 100 generations, ≥98% threshold, per-task activation (sin/tanh), flat NT [0.5, 0.5, 0.5, X] with correct ACh polarity (ACh=1.0 for XOR/AND/OR, ACh=0.0 for NAND/NOR), product aggregation, 30 seeds.
**Purpose**: Isolate gain/bias modulation (DA, 5HT, NE) from output inversion (ACh). Tests whether inversion alone suffices or if gain/bias differentiation is structurally necessary.

### Result: 0/30 converged (0%)

All 30 seeds show identical zero-variance pattern across all 100 generations:
- XOR: 1.000 (every seed, every generation)
- AND: 0.750 (every seed, every generation)
- OR: 0.750 (every seed, every generation)
- NAND: 0.750 (every seed, every generation)
- NOR: 0.750 (every seed, every generation)

### Key Finding

**Gain/bias modulation is structurally necessary for threshold differentiation.** With ACh polarity alone:
- XOR solves via sin activation (no modulation needed for parity)
- All threshold tasks plateau at 75% (3/4 patterns correct). The network finds ONE threshold that gets 3 of 4 Boolean patterns right, but cannot differentiate AND (θ≈1.5) from OR (θ≈0.5)
- No evolution occurs. Fitness is constant from generation 0

This creates a clean mechanistic hierarchy:
- **No modulation** (flat NT, ACh=1.0 for all): 50% on threshold tasks (chance)
- **Inversion only** (flat NT, correct ACh): 75% on threshold tasks (3/4 patterns)
- **Full modulation** (task-specific NT, correct ACh): 100% (all patterns)

### Data Location
`papers/emr-neuromodulation/results/strengthening/honest_neuromod_ablation/`

**Script**: `experiments/neuromodulation/run_honest_neuromod_ablation.py`

---

## 18. Continuous Domain (n=30)

**Config**: 3 regression tasks × 3 activation conditions × 30 seeds = 90 runs. Standalone direct-encoded networks (not EMR-HyperNEAT). Tasks: periodic (y=sin(2πx)), monotonic (y=tanh(3x−1.5)), step (y=𝟙[x>0.5]), all normalized to [0,1]. (μ+λ)-ES, pop=200, 200 gens, fitness = 1−MSE, threshold ≥0.95.

### Results

| Condition | Converged | Rate | Median Gen | Range |
|-----------|-----------|------|-----------|-------|
| uniform_tanh | 30/30 | 100% | 15 | 0–68 |
| per_task | 13/30 | 43.3% | 76 | 29–175 |
| uniform_sin | 2/30 | 6.7% | 118 | 82–153 |

### Per-Task Failure Analysis (unconverged seeds)

| Condition | Periodic Failures | Monotonic Failures | Step Failures |
|-----------|-------------------|-------------------|---------------|
| per_task (17 failures) | 12 (70.6%) | 2 (11.8%) | 7 (41.2%) |

The per-task condition's bottleneck is predominantly the periodic task (12/17 failures, 70.6%), suggesting that even when sin is correctly assigned to its matching task, simultaneous optimization with tanh-driven tasks remains challenging.

### Key Finding

**Inverts the Boolean finding.** In continuous regression:
- Uniform tanh is BEST (30/30, median 15 gen, fastest convergence)
- Per-task activation HURTS (only 13/30, 43.3%)
- Uniform sin is WORST (2/30, 6.7%, rugged fitness landscape for gradient-free search)

The Boolean barrier is structural impossibility specific to discrete parity patterns, NOT a general function approximation limitation. Universal approximation means tanh neurons CAN fit smooth continuous functions via superposition. Sin creates rugged fitness landscapes: small weight perturbations cause large oscillatory output swings, making NEAT optimization harder. Tanh dominates because continuous regression on smooth targets is a fundamentally different computational problem.

### Data Location
`papers/emr-neuromodulation/results/continuous_analogue/`
- 90 per-seed files: `uniform_tanh_seed*.json`, `uniform_sin_seed*.json`, `per_task_seed*.json`
- Summary: `continuous_analogue_20260227_235235.json`

**Script**: `experiments/neuromodulation/run_continuous_analogue.py`

---

## 19. Multi-Head Architectural Control (E17)

**Config**: Standalone direct-encoded MLPs (not EMR-HyperNEAT). 5 Boolean tasks (XOR, AND, OR, NAND, NOR) learned simultaneously via 5 output heads (one per task). No neuromodulation. N_HIDDEN=10, tanh hidden, sigmoid output. 2 architectures × 2 optimizers × 30 seeds = 120 runs.

### Architectures
- **1-layer**: 2 → 10 (tanh) → 5 (sigmoid)
- **2-layer**: 2 → 10 (tanh) → 10 (tanh) → 5 (sigmoid)

### Optimizers
- **(μ+λ)-ES**: pop=750, σ=0.3, μ=10% (75 parents), 100 generations
- **Adam**: lr=0.01, 500 steps, convergence at product accuracy ≥ 0.98

### Results

| Condition | Converged | Rate | Median |
|-----------|-----------|------|--------|
| 1-layer ES | 24/30 | 80.0% | 34 gen |
| 2-layer ES | 28/30 | 93.3% | 23 gen |
| 1-layer Adam | 30/30 | 100% | 78 steps |
| 2-layer Adam | 30/30 | 100% | 43 steps |

### Key Findings

1. **Even 1-layer multi-head tanh solves all 5 tasks** (80% ES, 100% Adam). The barrier is NOT tanh networks in general. It's the single-output neuromodulated architecture specifically.
2. **Multi-head eliminates the XOR ceiling**: Separate output weights per task allow different feature combinations from the same hidden layer. XOR gets its own output weights that can combine tanh features into alternating patterns.
3. **Depth helps ES but not Adam**: 2-layer ES improves from 80%→93.3%, while Adam achieves 100% regardless. The remaining ES failures are search failures, not architectural impossibility.
4. **Architecture-specific diagnosis**: The neuromodulation paper's barrier is specific to: (a) single shared output, (b) gain/bias modulation requiring task discrimination, (c) single hidden layer with uniform activation. Multi-head architectures sidestep (a) entirely but sacrifice substrate reuse.

### Data Location
`papers/emr-neuromodulation/results/e17_twolayer_control/`

**Script**: `experiments/neuromodulation/run_e17_twolayer_tanh_control.py`

---

## 20. Multi-Layer Neuromodulated MLP (E18)

**Config**: Direct-encoded MLPs with neuromodulation, 1/2/3 hidden layers (10 neurons each), Schema B NT vectors, s=5.0, (μ+λ)-ES (pop=750, μ=75, σ=0.3), 100 generations, ≥98% threshold, 30 seeds per condition.
**Purpose**: Test whether multi-layer depth overcomes the 75% barrier in neuromodulated single-output architectures. Resolves limitation (8).

### Results

| Condition | Converged | Rate | Median Gen | 95% CI | Range | Params |
|-----------|-----------|------|-----------|--------|-------|--------|
| 1-layer tanh | 23/30 | 76.7% | 17 | [11, 24] | 4–86 | 81 |
| 2-layer tanh | 30/30 | 100% | 12 | [10, 14] | 2–22 | 231 |
| 3-layer tanh | 30/30 | 100% | 13 | [12, 14.5] | 4–25 | 381 |
| 1-layer pertask | 30/30 | 100% | 3 | [2, 3] | 1–5 | 81 |
| 2-layer pertask | 30/30 | 100% | 6 | [5, 7] | 0–12 | 231 |
| 3-layer pertask | 30/30 | 100% | 11 | [9, 11] | 0–16 | 381 |

### Statistical Tests

- **1L vs 2L tanh**: Fisher's exact p=0.011 (barrier eliminated by depth)
- **Pertask 1L vs tanh 2L speed**: Mann-Whitney U=40, p=5.65e-10 (pertask 4× faster)
- **Parameter overhead**: 81 (1-layer) vs 231 (2-layer) = 2.9× more parameters

### Key Findings

1. **Barrier is depth-specific**: 2+ layers = 100% convergence with uniform tanh + neuromodulation
2. **1-layer direct encoding ≠ 0%**: 76.7% success vs 0% under indirect encoding. CPPN-based encoding imposes additional constraints
3. **Per-task activation remains superior**: 4× faster (median 3 vs 12), 2.9× fewer parameters, works at simplest architecture
4. **Resolves limitation (8)**: Multi-layer neuromodulated single-output architectures DO overcome the barrier

### Data Location
`papers/emr-neuromodulation/results/e18_multilayer_neuromod/`

**Script**: `experiments/neuromodulation/run_e18_multilayer_neuromod_control.py`

---

## 21. Higher-Dimensional NT Vectors (E19)

**Config**: Direct-encoded neuromodulated MLPs (1 hidden layer, 10 neurons), per-task activation (sin for XOR/XNOR, tanh for threshold), (μ+λ)-ES (pop=750, μ=75, σ=0.3), 100 generations, ≥98% threshold, 30 seeds per condition.
**Purpose**: Test whether increasing NT dimensionality resolves the 7-task cliff. 4 conditions × 30 seeds = 120 runs.

### Conditions

- **4D 7-task pertask**: Original 4D NT, 7 tasks (XOR, AND, OR, NAND, NOR, XNOR, IMPLY)
- **6D 7-task pertask**: 6D NT (3 modulatory + 2 extra + ACh), 7 tasks
- **8D 8-task pertask**: 8D NT, 8 tasks (+ NIMPLY)
- **6D 7-task tanh**: 6D NT, 7 tasks, uniform tanh (no per-task activation)

### Results

| Condition | Converged | Rate | Median Gen | 95% CI | Range | Bottleneck |
|-----------|-----------|------|-----------|--------|-------|------------|
| 4D 7-task pertask | 7/30 | 23.3% | 42 | [24, 70] | 14–87 | IMPLY (23/23) |
| 6D 7-task pertask | 7/30 | 23.3% | 39 | [20, 59] | 13–80 | IMPLY (23/23) |
| 8D 8-task pertask | 6/30 | 20.0% | 62 | [19, 82] | 17–84 | IMPLY+NIMPLY (24/24) |
| 6D 7-task tanh | 3/30 | 10.0% | 52 | [37, 70] | 37–70 | IMPLY(21), XOR(18), XNOR(18) |

### Statistical Tests

- **4D vs 6D**: Fisher's exact p=1.000 (identical rates)
- **4D vs 8D**: Fisher's exact p=1.000 (no improvement)
- **6D pertask vs 6D tanh**: Fisher's exact p=0.299 (per-task activation helps but not significantly at 7-task)

### IMPLY Barrier Analysis

IMPLY plateaus at exactly 0.75 fitness in ALL failed runs, the same ceiling as XOR under monotonic activation. IMPLY ([1,1,0,1]) is the only asymmetric 2-input Boolean function, requiring directional discrimination ((0,1)→1 but (1,0)→0) that symmetric gain/bias/gating modulation cannot encode.

### Key Findings

1. **NT dimensionality is NOT the bottleneck**: 4D = 6D = 23.3% (Fisher p=1.000)
2. **IMPLY is the sole barrier at 7 tasks**: 23/23 failures in both 4D and 6D
3. **NIMPLY adds a second asymmetric barrier at 8 tasks**: 24/24 failures each
4. **Direct encoding partially overcomes indirect encoding barrier**: 23.3% vs 0% at 7-task
5. **6D tanh: XOR/XNOR re-emerge**: Without per-task activation, XOR/XNOR become barriers again (18/27 failures each) alongside IMPLY
6. **Reframes scaling limit**: Task-structural incompatibility with symmetric modulation, not NT space saturation

### Data Location
`papers/emr-neuromodulation/results/e19_highdim_nt/`

**Script**: `experiments/neuromodulation/run_e19_highdim_nt.py`

---

## 22. Higher-Arity Capacity Control: E22b

**Config**: Direct-encoded neuromodulated MLPs (N_HIDDEN=40, doubled from E22's 20), 4-input Boolean tasks (Parity-4, AND-4, OR-4; 16 patterns), (μ+λ)-ES (pop=750, μ=75, σ=0.3), 200 generations, ≥98% threshold, Schema B NT vectors, s=5.0, 30 seeds per condition.
**Purpose**: Test whether the 1-layer tanh barrier at 4-input is capacity-limited. If doubling hidden neurons resolves it, the barrier is capacity-driven; if not, it's representational. 4 conditions × 30 seeds = 120 runs.

### Conditions

- **1layer_3task_uniform_tanh**: 1 hidden layer, 40 neurons, uniform tanh activation
- **1layer_3task_pertask**: 1 hidden layer, 40 neurons, per-task activation (sin for Parity-4, tanh for AND-4/OR-4)
- **2layer_3task_uniform_tanh**: 2 hidden layers, 40 neurons each, uniform tanh
- **1layer_3task_uniform_sin**: 1 hidden layer, 40 neurons, uniform sin activation

### Results

| Condition | Converged | Rate | E22 (N=20) | Median Gen | Range | Parity-4 Avg (failed) |
|-----------|-----------|------|-----------|-----------|-------|----------------------|
| 1L uniform tanh | 0/30 | 0.0% | 0.0% | N/A | N/A | 0.77 |
| 1L per-task | 3/30 | 10.0% | 13.3% | 170 | 107–175 | 0.93 |
| 2L uniform tanh | 23/30 | 76.7% | 43.3% | 162 | 113–192 | 1.00 |
| 1L uniform sin | 0/30 | 0.0% | 3.3% | N/A | N/A | 0.85 |

### Per-Task Breakdown (2L uniform tanh, N=40)

| Task | Solved | Rate |
|------|--------|------|
| Parity-4 | 30/30 | 100% |
| AND-4 | 24/30 | 80% |
| OR-4 | 28/30 | 93.3% |

Parity-4 bottleneck eliminated at 2L with increased capacity: all 30 seeds solve Parity-4. Remaining 7 failures are due to AND-4 (6 failures) and OR-4 (2 failures).

### Statistical Tests

- **1L tanh N=20 vs N=40**: Fisher's exact p=1.000 (both 0/30, capacity irrelevant)
- **1L pertask N=20 vs N=40**: Fisher's exact p=1.000 (4/30 vs 3/30, no difference)
- **2L tanh N=20 vs N=40**: Fisher's exact p=0.017 (13/30 vs 23/30, significant improvement)
- **1L sin N=20 vs N=40**: Fisher's exact p=1.000 (1/30 vs 0/30, no difference)

### Key Findings

1. **Barrier is representational, NOT capacity-limited**: 1L tanh 0/30 at both 20 and 40 neurons
2. **Depth + capacity interact**: 2L tanh improves 43.3%→76.7% with doubled neurons (p=0.017)
3. **Parity-4 bottleneck shifts**: At 2L with 40 neurons, Parity-4 is solved 30/30; AND-4 and OR-4 become the new bottlenecks
4. **Per-task and sin conditions unaffected**: More capacity without depth doesn't help search
5. **Failed Parity-4 accuracy identical**: 0.77 avg at both N=20 and N=40 for 1L tanh, same ceiling

### Data Location
`papers/emr-neuromodulation/results/e22b_higher_arity_capacity/`

---

## 23. Frozen Substrate Generalization: E24

**Config**: Direct-encoded neuromodulated MLPs (1 hidden layer, 10 neurons), (μ+λ)-ES (pop=750, μ=75, σ=0.3), Schema B NT vectors, s=5.0, 30 seeds per condition.
**Purpose**: Test whether weights trained on threshold tasks (AND/OR/NAND/NOR) with uniform tanh can solve XOR when activation is switched to sin. If so, the barrier is at test-time configuration; if not, it's in the training dynamics. 4 conditions × 30 seeds = 120 runs.

### Conditions

- **frozen_threshold_xor_eval**: Train on 4 threshold tasks (uniform tanh) → freeze weights → evaluate all 5 tasks with per-task activation (sin for XOR)
- **random_xor_eval**: Evaluate XOR with sin activation on random (untrained) population, baseline for what sin can achieve without threshold training
- **retrained_5task**: Train on 4 threshold tasks → retrain all 5 tasks with per-task activation (unfrozen)
- **from_scratch_5task**: Train all 5 tasks from random initialization with per-task activation

### Results

| Condition | Converged | Rate | Median Gen | Range | Key Metric |
|-----------|-----------|------|-----------|-------|------------|
| frozen_threshold→XOR | 0/30 | 0.0% | N/A | N/A | XOR acc=0.500 (chance) |
| random_init→XOR | 30/30 | 100% | N/A | N/A | best_acc=1.000 |
| retrained 5-task | 30/30 | 100% | 2 | 1–5 | Full convergence |
| from_scratch 5-task | 30/30 | 100% | 3 | 1–5 | Full convergence |

### Key Findings

1. **Frozen threshold weights provide ZERO XOR transfer**: 0/30 XOR solved, mean accuracy exactly 0.500 (chance = 2/4 patterns correct)
2. **Barrier is in training dynamics, not evaluation**: Tanh-trained weights learn monotonic decision boundaries incompatible with parity
3. **Random weights + sin = 30/30 XOR**: Confirms sin is sufficient for XOR; the frozen weights are the problem
4. **Retraining converges instantly**: Median 2 gen (retrained) vs median 3 gen (from scratch). Threshold training neither helps nor hurts
5. **Threshold tasks unaffected by freezing**: All 4 threshold tasks remain at 100% accuracy with frozen weights
6. **Switching activation post-hoc cannot fix the barrier**: The representational mismatch is baked into the learned weight structure

### Interpretation

This experiment closes the loop on the activation barrier mechanism. The barrier operates at three levels, all now confirmed:
- **Representational** (E22b): More capacity doesn't help. The function class is wrong
- **Temporal** (E24): The barrier is during training, not at evaluation. Switching activation post-hoc doesn't work
- **Structural** (E18): Depth overcomes it by providing intermediate features, the only parametric solution

### Data Location
`papers/emr-neuromodulation/results/e24_frozen_substrate/`

---

## 24. Multi-Layer Neuromod Extension (E18 ext): 4+5 Layer

**Config**: Direct-encoded neuromodulated MLPs (4 and 5 hidden layers, 10 neurons each), (μ+λ)-ES (pop=750, μ=75, σ=0.3), Schema B NT vectors, s=5.0, 100 gen. Two conditions per depth: uniform tanh, per-task activation. 2 depths × 2 conditions × 30 seeds = 120 runs (extending E18 from 180 to 300 total).

### Results

| Condition | Converged | Rate | Median Gen | Range |
|-----------|-----------|------|-----------|-------|
| 4-layer uniform tanh | 30/30 | 100% | 18 | 6–26 |
| 4-layer per-task | 30/30 | 100% | 14 | 9–22 |
| 5-layer uniform tanh | 30/30 | 100% | 20 | 10–52 |
| 5-layer per-task | 30/30 | 100% | 15 | 6–27 |

### Full E18 Depth Progression (1–5 Layers)

| Depth | Uniform Tanh | Per-Task | Uniform Median | Per-Task Median |
|-------|-------------|----------|----------------|-----------------|
| 1-layer | 23/30 (76.7%) | 30/30 (100%) | 17 | 3 |
| 2-layer | 30/30 (100%) | 30/30 (100%) | 12 | 6 |
| 3-layer | 30/30 (100%) | 30/30 (100%) | 13 | 11 |
| 4-layer | 30/30 (100%) | 30/30 (100%) | 18 | 14 |
| 5-layer | 30/30 (100%) | 30/30 (100%) | 20 | 15 |

### Key Findings

1. **No degradation through 5 layers**: All 4L and 5L conditions achieve 100% convergence
2. **Convergence slows modestly**: ~3–5 generations per added layer
3. **Per-task advantage consistent**: Faster convergence at every depth (4L: 14 vs 18, 5L: 15 vs 20)
4. **Diminishing returns beyond 2–3 layers**: 2-layer already resolves both XOR and IMPLY barriers

### Data Location
`papers/emr-neuromodulation/results/e18_multilayer_neuromod/`

---

## 25. Higher Arity 5-Input (E25)

**Config**: Direct-encoded neuromodulated MLPs (1 hidden layer 25 neurons or 2 hidden layers 25+25), (μ+λ)-ES (pop=750, μ=75, σ=0.3), Schema B NT vectors, s=5.0, 200 gen. Tasks: Parity-5 (32 input patterns), AND-5, OR-5. 4 conditions × 30 seeds = 120 runs.

### Results

| Condition | Converged | Rate | Median Gen | Key Detail |
|-----------|-----------|------|-----------|------------|
| 1L uniform tanh | 0/30 | 0.0% | N/A | Parity-5 avg acc 0.765 |
| 1L per-task | 25/30 | 83.3% | 66 | Most parameter-efficient (276 params) |
| 1L uniform sin | 19/30 | 63.3% | 93 | Sin drops from 100% (2-input) |
| 2L uniform tanh | 18/30 | 60.0% | 129.5 | 1,026 params |

### Key Findings

1. **Monotonic barrier absolute at 5-input**: 0/30 for uniform tanh
2. **Per-task activation most efficient**: 83.3% at 276 params vs 60% at 1,026 params (2-layer)
3. **Sin effectiveness drops with arity**: 100% (2-input) → 63.3% (5-input), gradient-free optimization harder at higher arity
4. **Genuine difficulty increase**: Even best condition (per-task) drops from 100% (2-input) to 83.3% (5-input)

### Data Location
`papers/emr-neuromodulation/results/e25_higher_arity_5input/`

---

## 26. Indirect Encoding Depth (E26)

**Config**: CPPN-based indirect encoding (6 inputs → 20 hidden sin neurons → 1 output, ~161 CPPN parameters) generating substrate weights for 1-, 2-, and 3-layer architectures. Per-task activation, Schema B NT vectors, (μ+λ)-ES (pop=750, μ=75, σ=0.3), 200 gen. Tasks: XOR, AND, OR, NAND, NOR. 3 conditions × 30 seeds = 90 runs.

### Results

| Condition | Converged | Rate | Median Gen | Compression |
|-----------|-----------|------|-----------|-------------|
| 1-layer | 30/30 | 100% | 26 | 1.0× |
| 2-layer | 29/30 | 96.7% | 26 | 4.1× |
| 3-layer | 23/30 | 76.7% | 84 | 7.2× |

### Key Findings

1. **2-layer is the sweet spot**: 96.7% with 4.1× parameter compression and no convergence penalty vs 1-layer
2. **3-layer degrades**: 76.7% success, median jumps to 84 gen, failures plateau at 0.75 fitness
3. **AND/NAND bottleneck**: 71% of 3-layer failures are on AND or NAND tasks
4. **Real parameter compression**: CPPN generates increasingly more substrate parameters with depth while CPPN size stays constant

### Data Location
`papers/emr-neuromodulation/results/e26_indirect_multilayer/`

---

## 27. Task Scaling 2-Layer (E27)

**Config**: Direct-encoded neuromodulated MLPs (2 hidden layers, 10 neurons each), (μ+λ)-ES (pop=750, μ=75, σ=0.3), Schema B-extended NT vectors, s=5.0, 200 gen. All 10 non-trivial 2-input Boolean functions. 4 conditions × 30 seeds = 120 runs.

### Results

| Condition | Converged | Rate | Median Gen |
|-----------|-----------|------|-----------|
| 2L 8-task per-task | 30/30 | 100% | 46.5 |
| 2L 8-task uniform tanh | 30/30 | 100% | 50.5 |
| 2L 10-task per-task | 9/30 | 30.0% | 125 |
| 2L 10-task uniform tanh | 13/30 | 43.3% | 126 |

### Key Findings

1. **Depth extends scaling**: 6 tasks (1-layer) → 8 tasks (2-layer) with 100% in both conditions
2. **10-task cliff**: New barrier at 10 tasks, 30.0–43.3%
3. **Implication family bottleneck**: All 6 symmetric tasks (XOR, AND, OR, NAND, NOR, XNOR) achieve 100%; the 4 asymmetric tasks (IMPLY, NIMPLY, CONVERSE, CONVERSE-NIMPLY) are sole failure points
4. **Per-task hurts at 10 tasks**: 30.0% vs 43.3% (p=0.422), optimization complexity outweighs representational benefit at extreme task counts

### Data Location
`papers/emr-neuromodulation/results/e27_task_scaling_2layer/`

---

## 28. 2D Synthetic Classification (E28)

**Config**: Direct-encoded neuromodulated MLPs, (μ+λ)-ES (pop=750, μ=75, σ=0.3), s=5.0, 200 gen. Tasks: concentric circles (non-linear boundary), linearly separable blobs, half-moons (200 data points each). Fitness = classification accuracy, threshold ≥ 0.95. 4 conditions × 30 seeds = 120 runs.

### Results

| Condition | Converged | Rate | Circles Acc | Linear Acc | Moons Acc |
|-----------|-----------|------|------------|-----------|-----------|
| 1L uniform tanh | 0/30 | 0.0% | 0.876 | 0.996 | 0.889 |
| 1L per-task | 0/30 | 0.0% | 0.888 | 0.990 | 0.871 |
| 1L uniform sin | 0/30 | 0.0% | 0.782 | 0.889 | 0.733 |
| 2L uniform tanh | 4/30 | 13.3% | 0.975 | 0.997 | 0.903 |

### Key Findings

1. **Per-task activation neutral-to-harmful for continuous 2D**: 0/30, unlike Boolean where it's transformative
2. **Sin catastrophically bad**: Lowest accuracy across all tasks. Rugged oscillatory landscape impedes gradient-free optimization
3. **Depth provides only path to improvement**: 2-layer tanh 13.3% (only condition with any convergence)
4. **Half-moon bottleneck**: Noise overlap near decision boundary prevents reaching 0.95 threshold
5. **Domain-dependent interaction**: Boolean tasks have structural impossibility barriers; continuous tasks have soft optimization barriers where depth matters more than activation choice

### Data Location
`papers/emr-neuromodulation/results/e28_synthetic_2d/`

---

## Extended-Generation Confirmation (10×): Indirect EMR-HyperNEAT (2026-06-07)

**Reviewer 4 (ALIFE 2026) request**: does the uniform-tanh barrier survive a 10× generation budget, or is it just insufficient runtime?

**Setup**: INDIRECT EMR-HyperNEAT (CPPN substrate, `multihead_palette_neuromodulation`, `palette_mode='uniform'`, product aggregation, Schema-B NT, Pop=750, max_depth=4), uniform tanh, 5-task (XOR+AND+OR+NAND+NOR), **1000 generations** (10× the 100-gen baseline), 30 seeds. Runner: `benchmark_extended_generations_indirect.py`. ~30 min/seed.

| Condition | Result | XOR | Min fitness |
|-----------|--------|-----|-------------|
| Indirect uniform tanh, 5-task, 1000 gens | **0/30 converged** | 0.750 (all seeds) | 0.750 (all seeds) |

**Finding**: the 75% XOR ceiling is unchanged at 10× the generation budget. Every one of the 30 seeds plateaus at exactly XOR=0.750 (clean HMR backend; the barrier is backend-agnostic; EMR also gives 0/30 XOR=0.75; see the EMR migration N=30 re-validation). Confirms a genuine evolutionary convergence barrier (for the (μ+λ)-ES tested), not insufficient runtime. `population: 750` verified in each JSON (the `pop_size` wiring bug does not affect this pipeline).

**Caveat (encoding)**: the DIRECT-encoded MLP control is a different, weaker regime: 2-task XOR+AND converges ~96.7% at 1000 gens (`benchmark_extended_generations.py`). Only the INDIRECT encoding exhibits the headline barrier; the extended-generation confirmation must use the indirect pipeline.

### Data Location
`papers/emr-neuromodulation/results/extended_generations_indirect/` (gitignored)

---

## HMR → EMR Migration Validation (2026-06-08)

The neuromod runner (`multihead_palette_neuromodulation.py`) was migrated from the HMR backend to the EMR backend (now `backend='emr'` by default; `backend='hmr'` reproduces the exact published baseline). The neuromod forward (Eqs 1–3) is the runner's own and unchanged; only the genome→substrate decode differs. EMR's decode initially used a first-input-weight approximation; wiring the `use_self_connection_query` flag (gated, default-off) makes it source receptors/base-gains from the CPPN self-connection query, the same source as HMR/the paper.

**N=30 re-validation under EMR** (`validate_emr_migration.py --n30`, 30 paper seeds, 100 gens, Pop=750):

| Condition | EMR result | Paper (HMR) |
|-----------|-----------|-------------|
| Per-task 5-task (per_head, product) | **30/30 = 100%** (median gen 10.5, all XOR=1.0) | 30/30, median 14 |
| Uniform-tanh 5-task (barrier) | **0/30**, XOR=0.75 on all 30 seeds | 0/30, XOR=0.750 |

EMR reproduces the headline conclusions and the exact 75% ceiling. Per-seed convergence gens differ from HMR (distinct code paths; not byte-identical), as expected.

**Indirect smoke test under EMR** (`smoke_test_emr_indirect.py`, 1 seed, 60 gens), every distinct indirect config runs and reproduces the paper's direction (**7/7**): per-task→100%, uniform-tanh→0.75 barrier, uniform-sin→converge, min-aggregation→converge, flat-NT neuromod ablation→fail (min=0.50; XOR solved but threshold tasks at chance without task-specific NT), 2-task per-task→converge, 2-task uniform→0.75 barrier.

**Scope:** the migration affects only the 18 INDIRECT scripts (they use the runner). The ~27 direct-encoded control scripts (monotonic ablation, Adam, oscillatory-3input, pop/topology sweeps) use separate code and are unaffected.
