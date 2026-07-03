"""Paper-validation tests.

Each test re-runs the *real* runner code from one of the published papers on a fixed
seed and asserts the paper's headline finding plus recorded golden values. The point is a
permanent, CI-checkable guarantee that the standalone repo still reproduces what the
papers claim.

Design notes
------------
* **Real paper code, not duplicated config.** The paper runners under
  ``papers/<paper>/scripts/`` expose single-seed functions that *return* a result and do
  NOT write to disk, so the tests import and call them directly. No dependence on the
  gitignored result JSON (the Zenodo data release) -- everything is recomputed fresh.

* **One batch subprocess per paper.** Different papers ship same-named helper modules with
  *different* contents (e.g. ``classification_problems.py`` differs between
  emr-dynamic-functions and emr-dynamic-functions-bio-inspired), so importing two runners
  in one interpreter would cross-contaminate ``sys.modules``. Each paper therefore runs in
  its own subprocess. To keep things fast (a JAX import dominates each subprocess), every
  paper runs *all* of its experiments in a single ``scope="class"`` fixture and the
  individual tests assert against the cached results. The base-EMR test uses the in-package
  ``conftest`` helpers and runs in-process.

* **Findings vs goldens.** The *finding* asserts (solved / converged / threshold) are the
  portable contract -- they survive tiny cross-platform float differences and ARE the paper
  claim. Exact goldens are asserted where the underlying value is a continuous deterministic
  fitness (dyn-func, base EMR); the ES accuracies (neuromod) are counts/200 that can shift a
  point across backends, so those use the finding + a band. Validated under act on emulated
  linux/amd64 -- the arm64-recorded goldens hold. Bit-identity itself is covered by
  ``test_golden.py``.

Markers: every test here is ``paper``; the bio tests are additionally ``slow``.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

# emr-hyperneat/ (repo root): tests -> emr_hyperneat -> repo
REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "papers"

# Tolerance for the continuous-fitness goldens. Loose enough to survive a different CPU
# backend, far tighter than any real regression or non-determinism (which move the value by
# >> this). Bit-identity is enforced elsewhere (test_golden.py).
GOLD_TOL = 1e-5

# Oscillatory activation indices, per the bio runner's own definition.
OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

pytestmark = pytest.mark.paper


def _run_paper(scripts_dir: Path, body: str, timeout: int = 900) -> dict:
    """Run ``body`` in a fresh interpreter with ``scripts_dir`` on ``sys.path``.

    ``body`` must assign a JSON-serialisable dict to a variable named ``out``. Returns the
    parsed dict. A fresh process guarantees no cross-paper module-name collisions and a
    clean JAX state.
    """
    code = (
        "import sys, json\n"
        # runner modules live in scripts/runners/; shared libraries at the scripts/ root
        f"sys.path.insert(0, {str(scripts_dir)!r})\n"
        f"sys.path.insert(0, {str(scripts_dir / 'runners')!r})\n"
        f"{body}\n"
        'print("PAPER_RESULT_JSON=" + json.dumps(out))\n'
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "TF_CPP_MIN_LOG_LEVEL": "3"}
    env.pop("PYTHONPATH", None)  # prove no reliance on the geenns env
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"paper subprocess failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PAPER_RESULT_JSON="):
            return json.loads(line[len("PAPER_RESULT_JSON="):])
    raise AssertionError(
        f"no result emitted.\nSTDOUT tail:\n{proc.stdout[-1500:]}\n"
        f"STDERR tail:\n{proc.stderr[-1500:]}"
    )


# ===========================================================================
# Dynamic functions (ALIFE): per-node activation evolution -- the oscillatory /
# monotonic divide on XOR. Oscillatory activations solve within a few generations;
# monotonic ones do not (an evolvability barrier, not strict impossibility -- tanh would
# eventually solve near gen 48); pure-linear identity is pinned at 3/4.
# ===========================================================================
_OSCILLATORY = ["sin", "osc_adapt", "resonator", "burst"]
_MONOTONIC = ["tanh", "sigmoid", "relu", "identity"]


@pytest.fixture(scope="class")
def dynfunc_results():
    body = (
        "import E2_xor_per_function as e2\n"
        "L = e2.ACTIVATION_LIST\n"
        "out = {}\n"
        "for fn in ['sin', 'osc_adapt', 'resonator', 'burst', "
        "'tanh', 'sigmoid', 'relu', 'identity']:\n"
        "    r = e2.run_single_trial(fn, L.index(fn), 42, max_generations=10)\n"
        "    out[fn] = {'solved': bool(r.solved), 'solved_gen': r.solved_gen, "
        "'best': float(r.best_fitness)}\n"
        # one deeper check: tanh still fails with 4x the budget the oscillatory ones needed
        "r = e2.run_single_trial('tanh', L.index('tanh'), 42, max_generations=40)\n"
        "out['tanh_mg40'] = {'solved': bool(r.solved), 'best': float(r.best_fitness)}\n"
    )
    return _run_paper(PAPERS / "emr-dynamic-functions" / "scripts", body)


class TestDynamicFunctionsPaper:
    @pytest.mark.parametrize("fn", _OSCILLATORY)
    def test_oscillatory_solves_xor(self, dynfunc_results, fn):
        r = dynfunc_results[fn]
        assert r["solved"] is True, f"oscillatory {fn} should solve XOR"
        assert r["solved_gen"] <= 10

    @pytest.mark.parametrize("fn", _MONOTONIC)
    def test_monotonic_does_not_solve_xor(self, dynfunc_results, fn):
        r = dynfunc_results[fn]
        assert r["solved"] is False, f"monotonic {fn} should not solve XOR in 10 gens"
        assert r["best"] < 0.95

    def test_sin_golden(self, dynfunc_results):
        assert abs(dynfunc_results["sin"]["best"] - 0.9999833106994629) < GOLD_TOL

    def test_identity_stuck_at_three_quarters(self, dynfunc_results):
        # pure-linear identity cannot represent XOR -> pinned at exactly 3/4
        assert abs(dynfunc_results["identity"]["best"] - 0.75) < 1e-6

    def test_tanh_unsolved_even_at_40_generations(self, dynfunc_results):
        r = dynfunc_results["tanh_mg40"]
        assert r["solved"] is False
        assert abs(r["best"] - 0.8749328851699829) < GOLD_TOL


# ===========================================================================
# Parity scaling (ALIFE): a monotonic activation does not merely struggle on XOR -- it
# COLLAPSES as task arity grows. The same tanh that eventually solves Parity-2 (XOR) around
# gen 48 cannot solve Parity-4 at all within 100 generations. (E1 measures the monotonic
# side; the paper's sin stays at 100% across all parities.)
# ===========================================================================
@pytest.fixture(scope="class")
def parity_scaling_results():
    body = (
        "import E1_parity_scaling as e1\n"
        "out = {}\n"
        "for nb in (2, 4):\n"
        "    r = e1.run_single_seed(nb, 42)\n"
        "    out[str(nb)] = {'solved': bool(r['solved']), "
        "'solved_gen': r['solved_gen'], 'best': float(r['best_fitness'])}\n"
    )
    return _run_paper(PAPERS / "emr-dynamic-functions" / "scripts", body)


class TestParityScalingPaper:
    def test_tanh_solves_parity2_but_slowly(self, parity_scaling_results):
        # tanh CAN solve XOR (Parity-2) -- but only after a long search (~48 gens), versus
        # the oscillatory activations that solve it in 1-6 generations
        r = parity_scaling_results["2"]
        assert r["solved"] is True
        assert r["solved_gen"] >= 30
        assert abs(r["best"] - 0.9823189377784729) < GOLD_TOL

    def test_tanh_collapses_at_parity4(self, parity_scaling_results):
        # ... and the same activation collapses entirely at Parity-4 -- never solves in 100 gens
        r = parity_scaling_results["4"]
        assert r["solved"] is False
        assert r["best"] < 0.95
        assert abs(r["best"] - 0.8664138317108154) < GOLD_TOL


# ===========================================================================
# Neuromodulation (ALIFE): multi-task barrier. Three ways escape it -- per-task activation
# selection, an oscillatory (sin) uniform activation, and added depth (2 layers) -- all
# converge on both tasks; a single uniform monotonic (tanh) activation in one layer cannot:
# the XOR-like task sticks near the 0.75 ceiling while the linear task still solves.
# ES accuracies are counts/200, so assert the finding + a band, not an exact golden.
# ===========================================================================
_CONVERGING_CONDITIONS = [
    "1layer_2task_pertask",
    "1layer_2task_uniform_sin",
    "2layer_2task_uniform_tanh",
]


@pytest.fixture(scope="class")
def neuromod_results():
    body = (
        "import benchmark_4class_xor as nm\n"
        "data = nm.generate_all_data(0)\n"
        "out = {}\n"
        "for cond in ['1layer_2task_pertask', '1layer_2task_uniform_sin', "
        "'2layer_2task_uniform_tanh', '1layer_2task_uniform_tanh']:\n"
        "    r = nm.run_es(cond, 0, data, pop_size=750, generations=15, verbose=False)\n"
        "    out[cond] = {'converged': bool(r['converged']), "
        "'min_acc': float(r['min_task_accuracy']), "
        "'per_task': {k: float(v) for k, v in r['per_task_fitness'].items()}}\n"
    )
    return _run_paper(PAPERS / "emr-neuromodulation" / "scripts", body)


class TestNeuromodulationPaper:
    @pytest.mark.parametrize("cond", _CONVERGING_CONDITIONS)
    def test_condition_escapes_barrier(self, neuromod_results, cond):
        # finding: per-task selection / oscillatory activation / depth each solve BOTH tasks
        r = neuromod_results[cond]
        assert r["converged"] is True, f"{cond} should converge"
        assert r["min_acc"] >= 0.90

    def test_uniform_tanh_barrier(self, neuromod_results):
        # finding: a single monotonic activation in one layer cannot serve both tasks --
        # the XOR-like task is stuck near the 0.75 ceiling, the linear task solves
        r = neuromod_results["1layer_2task_uniform_tanh"]
        assert r["converged"] is False
        assert r["per_task"]["xor_4cluster"] < 0.80
        assert r["per_task"]["linear_4cluster"] > 0.90
        assert 0.70 <= r["per_task"]["xor_4cluster"] <= 0.78  # band golden (~0.74)


# ===========================================================================
# Base EMR (GECCO): the GPU-speedup result is measured from committed data and cannot be
# re-run fresh in CI, so the representative checks are that the base EMR algorithm the
# paper measured (a) produces a known XOR trajectory and (b) is deterministic. In-process
# via conftest (no paper-script import -> no collision).
# ===========================================================================
class TestBaseEMRPaper:
    def test_base_emr_xor_golden(self):
        from conftest import (
            EMRHyperNEAT, create_base_config, run_quick_evolution, XORProblem,
        )
        result = run_quick_evolution(
            EMRHyperNEAT(), create_base_config(), XORProblem(), generations=3, seed=42,
        )
        assert result.error is None
        assert math.isfinite(result.best_fitness) and result.best_fitness > 0.0
        assert abs(result.best_fitness - 0.8140925765037537) < GOLD_TOL

    def test_base_emr_deterministic(self):
        from conftest import (
            EMRHyperNEAT, create_base_config, run_quick_evolution, XORProblem,
        )
        r1 = run_quick_evolution(
            EMRHyperNEAT(), create_base_config(), XORProblem(), generations=3, seed=42,
        )
        r2 = run_quick_evolution(
            EMRHyperNEAT(), create_base_config(), XORProblem(), generations=3, seed=42,
        )
        assert r1.error is None and r2.error is None
        # same seed + config -> bit-identical best-fitness trajectory
        assert r1.fitness_history == r2.fitness_history


# ===========================================================================
# Bio-inspired palette evolution (PPSN): a timescale-compatible strategy (circadian)
# discovers AND retains an oscillatory activation even when seeded from a purely
# non-oscillatory palette [band_pass(16), integrate(17)] on Parity-4. Slow (HMR on
# Parity-4); excluded from the fast gate, run on demand. All three checks share one run.
# ===========================================================================
@pytest.mark.slow
class TestBioInspiredPaper:
    @pytest.fixture(scope="class")
    def bio_result(self):
        body = (
            "import benchmark_bio_strategy_fixed_palette as bio\n"
            "r = bio.run_single_trial(bio.STRATEGIES['circadian_rhythm_dual'], "
            "seed=42, pop_size=40, max_gens=6)\n"
            "out = {'has_oscillatory': bool(r['has_oscillatory']), "
            "'osc_discovered_gen': r['osc_discovered_gen'], "
            "'n_osc': len(r['oscillatory_functions']), "
            "'best': float(r['best_fitness']), "
            "'initial_palette': r['initial_act_palette']}\n"
        )
        return _run_paper(
            PAPERS / "emr-dynamic-functions-bio-inspired" / "scripts", body, timeout=600,
        )

    def test_starts_from_non_oscillatory_palette(self, bio_result):
        # the experiment's premise: seeded from a palette with NO oscillatory function
        assert bio_result["initial_palette"] == [16, 17]
        assert all(i not in OSCILLATORY_INDICES for i in bio_result["initial_palette"])

    def test_circadian_discovers_oscillatory(self, bio_result):
        # finding: ... yet it discovers and retains an oscillatory activation
        assert bio_result["has_oscillatory"] is True
        assert bio_result["osc_discovered_gen"] is not None
        assert bio_result["n_osc"] >= 1

    def test_best_fitness_golden(self, bio_result):
        assert math.isfinite(bio_result["best"]) and 0.0 <= bio_result["best"] <= 1.0
        assert abs(bio_result["best"] - 0.7666119933128357) < 1e-3  # band (HMR + palette)


# ===========================================================================
# EMR substrate structure (GECCO): the eager multi-resolution substrate -- the mechanism
# behind the GPU speedup -- is a FIXED hierarchical lattice (4^(l+1) cells at level l) that
# post-hoc variance masking prunes to a SPARSE active set. In-process (public emr_hyperneat
# API, no paper-script import). FAST.
# ===========================================================================
class TestEMRSubstrateStructure:
    def test_grid_size_scales_geometrically_with_depth(self):
        # total candidate positions = sum_{l=0..D} 4^(l+1)
        from emr_hyperneat.emrhyperneat_base import get_hierarchical_grid
        assert get_hierarchical_grid(2).total_positions == 84
        assert get_hierarchical_grid(3).total_positions == 340
        assert get_hierarchical_grid(4).total_positions == 1364
        assert get_hierarchical_grid(5).total_positions == 5460

    def test_variance_masking_yields_sparse_substrate(self):
        from conftest import EMRHyperNEAT, create_base_config, XORProblem
        algo = EMRHyperNEAT()
        cfg = algo.create_config(create_base_config())  # max_depth 2 -> 84 candidate positions
        state = algo.initialize(cfg, XORProblem(), seed=42)
        _, m = algo.run_generation(state, XORProblem())
        cm = m.custom_metrics
        assert cm["total_positions"] == 84
        # the eager grid is masked down to a sparse active subset
        assert 1 <= cm["avg_hidden_nodes"] <= cm["total_positions"]
        assert cm["max_hidden_nodes"] <= cm["total_positions"]
        assert 0.0 < cm["position_utilization"] < 1.0
        assert abs(cm["position_utilization"]
                   - cm["avg_hidden_nodes"] / cm["total_positions"]) < 1e-6


# ===========================================================================
# Oscillatory CLASS escapes the multi-task barrier (ALIFE): the barrier escape generalizes
# beyond sin -- cos and GCU (x*cos x) also solve the 5-task neuromodulation problem. FAST.
# ===========================================================================
_OSC_CLASS = ["uniform_cos", "uniform_gcu"]


@pytest.fixture(scope="class")
def alt_oscillatory_results():
    body = (
        "import benchmark_alt_oscillatory as ao\n"
        "out = {}\n"
        "for cond in ('uniform_cos', 'uniform_gcu'):\n"
        "    r = ao.run_es(cond, 0, generations=15, verbose=False)\n"
        "    out[cond] = {'converged': bool(r['converged']), 'conv_gen': r['convergence_gen']}\n"
    )
    return _run_paper(PAPERS / "emr-neuromodulation" / "scripts", body)


class TestOscillatoryClassPaper:
    @pytest.mark.parametrize("cond", _OSC_CLASS)
    def test_oscillatory_class_escapes_barrier(self, alt_oscillatory_results, cond):
        r = alt_oscillatory_results[cond]
        assert r["converged"] is True, f"{cond} should escape the multi-task barrier"
        assert r["conv_gen"] is not None and r["conv_gen"] <= 10


# ===========================================================================
# Gradient-baseline reversal (ALIFE), SLOW. Under direct gradient descent (Adam MLP) the
# neuroevolution oscillatory advantage REVERSES: tanh solves Parity-8 while sin fails. So the
# advantage is specific to indirect-encoding neuroevolution, not a universal "sin is good for
# parity" effect. (train_mlp is pure optax, no module-name collision -- but unjitted/slow.)
# ===========================================================================
@pytest.mark.slow
class TestGradientBaselinePaper:
    @pytest.fixture(scope="class")
    def gradient_results(self):
        body = (
            "import E6_gradient_baseline as e6\n"
            "raw = {\n"
            "  'tanh_p4': e6.train_mlp(42, 4, 16, 'tanh', 0.01),\n"
            "  'sin_p4':  e6.train_mlp(42, 4, 16, 'sin', 0.01),\n"
            "  'tanh_p8': e6.train_mlp(42, 8, 32, 'tanh', 0.01, max_epochs=2000),\n"
            "  'sin_p8':  e6.train_mlp(42, 8, 32, 'sin', 0.01, max_epochs=500),\n"
            "}\n"
            "out = {k: {'solved': bool(v['solved']), 'acc': float(v['best_accuracy'])} "
            "for k, v in raw.items()}\n"
        )
        return _run_paper(PAPERS / "emr-dynamic-functions" / "scripts", body)

    def test_parity4_both_activations_solve(self, gradient_results):
        # at low arity, gradient descent solves with either activation
        assert gradient_results["tanh_p4"]["solved"] is True
        assert gradient_results["sin_p4"]["solved"] is True

    def test_parity8_gradient_reversal(self, gradient_results):
        # tanh SOLVES Parity-8 under gradient descent ...
        assert gradient_results["tanh_p8"]["solved"] is True
        assert gradient_results["tanh_p8"]["acc"] >= 0.95
        # ... while sin FAILS -- the opposite of the neuroevolution result
        assert gradient_results["sin_p8"]["solved"] is False
        assert gradient_results["sin_p8"]["acc"] < 0.60


# ===========================================================================
# Monotonic-CLASS multi-task barrier (ALIFE), SLOW. A pure-linear (identity) activation solves
# the linearly-separable tasks (AND/OR/NAND/NOR) but leaves XOR pinned at the 0.75 ceiling, so
# it never converges on the 5-task set -- the barrier is the monotonic class, not just tanh.
# ===========================================================================
@pytest.mark.slow
class TestMonotonicClassBarrierPaper:
    @pytest.fixture(scope="class")
    def identity_result(self):
        body = (
            "import benchmark_monotonic_ablation as ma\n"
            "r = ma.run_es('uniform_identity', 0, generations=15, verbose=False)\n"
            "pt = r.get('per_task_fitness') or r.get('per_task')\n"
            "out = {'converged': bool(r['converged']), "
            "'per_task': {k: float(v) for k, v in pt.items()}}\n"
        )
        return _run_paper(PAPERS / "emr-neuromodulation" / "scripts", body)

    def test_identity_barrier_is_xor_specific(self, identity_result):
        r = identity_result
        assert r["converged"] is False
        assert r["per_task"]["xor"] < 0.80  # XOR stuck at the 0.75 ceiling
        for t in ("and", "or", "nand", "nor"):  # ... the linearly-separable tasks still solve
            assert r["per_task"][t] >= 0.95


# ===========================================================================
# Aggregation oracle (PPSN), SLOW. A fixed global 'min' aggregation cannot substitute for
# palette evolution: monotonic activations + min-aggregation still fail Parity-4.
# ===========================================================================
@pytest.mark.slow
class TestAggregationOraclePaper:
    @pytest.fixture(scope="class")
    def agg_result(self):
        body = (
            "import benchmark_bio_agg_oracle as ao\n"
            "r = ao.run_single_trial('monotonic_min', 42, pop_size=150, max_gens=15)\n"
            "out = {'solved': bool(r['solved']), 'best': float(r['best_fitness'])}\n"
        )
        return _run_paper(
            PAPERS / "emr-dynamic-functions-bio-inspired" / "scripts", body, timeout=600,
        )

    def test_min_aggregation_does_not_rescue_monotonic(self, agg_result):
        assert agg_result["solved"] is False
        assert agg_result["best"] < 0.95


# ===========================================================================
# Single-task control (ALIFE), SLOW -- the heaviest test (~6 min CPU). The multi-task barrier is
# INTERFERENCE, not an inability of monotonic activations to represent XOR: the same uniform-tanh
# substrate that FAILS the 2-task problem (TestNeuromodulationPaper.test_uniform_tanh_barrier)
# SOLVES a single XOR on its own through the indirect EMR pipeline.
# ===========================================================================
@pytest.mark.slow
class TestSingleTaskControlPaper:
    @pytest.fixture(scope="class")
    def single_task_result(self):
        body = (
            "import multihead_palette_neuromodulation as mh\n"
            "r = mh.run_multihead_palette_experiment(task_names=['xor'], palette_mode='uniform', "
            "blend_mode='fixed', aggregation='product', seed=0, success_threshold=0.90, "
            "verbose=False)\n"
            "out = {'converged': bool(r.converged)}\n"
        )
        return _run_paper(
            PAPERS / "emr-neuromodulation" / "scripts", body, timeout=3000,
        )

    def test_uniform_tanh_solves_single_xor(self, single_task_result):
        # uniform tanh SOLVES a single XOR -> the 2-task collapse is interference, not inability
        assert single_task_result["converged"] is True
