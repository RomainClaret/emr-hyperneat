#!/usr/bin/env python3
"""Adversarial recompute of every numeric claim in ppsn_main.tex from raw JSON.

Recomputes solve rate, median-gen-to-solve, oscillatory presence, and sin
discovery directly from the per-run result lists (NEVER from the stored summary,
which uses inconsistent scaling across files). Prints PAPER vs RECOMPUTED so any
mismatch is obvious.
"""
import json
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results"

# Canonical index -> name (ACTIVATION_LIST in emrhyperneat.py)
ACT = ['tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu',
       'softplus', 'rs_adapt', 'fs_fast', 'lts_low', 'burst', 'resonator',
       'osc_adapt', 'gain_mod', 'receptive', 'band_pass', 'integrate']
OSC = {4, 11, 12, 13, 15}  # sin, burst, resonator, osc_adapt, receptive
SIN = 4


def act_palette(run):
    """Return the activation palette for a run (act-only or dual)."""
    if 'final_palette' in run and run['final_palette'] is not None:
        return run['final_palette']
    if 'final_act_palette' in run and run['final_act_palette'] is not None:
        return run['final_act_palette']
    return None


def is_dual(run):
    return 'final_agg_palette' in run and run.get('final_agg_palette') is not None


def load(path):
    d = json.load(open(RES / path))
    if isinstance(d, dict) and 'results' in d:
        return d['results']
    if isinstance(d, dict) and 'trials' in d:  # oracle schema (fixed-palette runs)
        return d['trials']
    if isinstance(d, list):
        return d
    raise ValueError(f"no results list in {path}")


def metrics(path):
    runs = load(path)
    n = len(runs)
    solved = [r for r in runs if r.get('solved')]
    ns = len(solved)
    rate = 100.0 * ns / n if n else float('nan')
    gens = [r['solved_gen'] for r in solved if r.get('solved_gen') is not None]
    medgen = statistics.median(gens) if gens else None
    # oscillatory presence among SOLVED runs (palette contains any osc index)
    osc_solved = sum(1 for r in solved
                     if act_palette(r) and any(i in OSC for i in act_palette(r)))
    osc_pct = 100.0 * osc_solved / ns if ns else float('nan')
    # sin discovery: fraction of SOLVED with pure sin (idx 4) in palette
    sin_solved = sum(1 for r in solved
                     if act_palette(r) and SIN in act_palette(r))
    sin_solved_pct = 100.0 * sin_solved / ns if ns else float('nan')
    # sin discovery over ALL runs (matches some summary fields)
    sin_all = sum(1 for r in runs if act_palette(r) and SIN in act_palette(r))
    sin_all_pct = 100.0 * sin_all / n if n else float('nan')
    dual = any(is_dual(r) for r in runs)
    return dict(path=path, n=n, ns=ns, rate=rate, medgen=medgen,
                osc_pct=osc_pct, osc_solved=osc_solved,
                sin_solved_pct=sin_solved_pct, sin_all_pct=sin_all_pct,
                dual=dual)


def show(label, path, paper=""):
    try:
        m = metrics(path)
    except Exception as e:
        print(f"  !! {label}: ERROR {e}")
        return None
    cfg = "DUAL" if m['dual'] else "act-only"
    print(f"  {label:28s} [{cfg:8s}] n={m['n']:3d} solved={m['ns']:3d} "
          f"rate={m['rate']:5.1f}% medgen={str(m['medgen']):>5s} "
          f"osc={m['osc_pct']:5.1f}% sinSolved={m['sin_solved_pct']:5.1f}% "
          f"sinAll={m['sin_all_pct']:5.1f}%   <paper: {paper}>")
    return m


print("=" * 110)
print("TABLE 1  single-task Parity-4 activation-only (single_task/, seeds 42-71)")
print("  caption: SinDisc=fraction of SOLVED with pure sin; Osc%=fraction SOLVED with any osc")
print("=" * 110)
show("Circadian", "single_task/circadian_rhythm.json", "97% g20 sin93 osc100")
show("Hebbian", "single_task/hebbian.json", "90% g16 sin100 osc100")
show("CriticalPeriod", "single_task/critical_period.json", "90% g31 sin100 osc100")
show("STDP", "single_task/stdp.json", "90% g34 sin100 osc100")
show("Metaplastic", "single_task/metaplastic.json", "80% g24 sin100 osc100")
show("Predator-Prey", "single_task/predator_prey.json", "77% g39 sin0 osc100")
show("Baseline", "single_task/baseline.json", "70% g59 sin37 osc100")
show("Neurogenesis", "single_task/adult_neurogenesis.json", "53% g53.5 sin37 osc100")

print()
print("=" * 110)
print("TABLE 2  tab:n60 N=60 activation-only = single_task (42-71) + single_task_actonly_72_101 (72-101)")
print("=" * 110)


def merge2(label, f1, f2, paper=""):
    try:
        runs = load(f1) + load(f2)
    except Exception as e:
        print(f"  !! {label}: ERROR {e}")
        return
    n = len(runs)
    ns = sum(1 for r in runs if r.get('solved'))
    dual = any(is_dual(r) for r in runs)
    cfg = "DUAL" if dual else "act-only"
    print(f"  {label:28s} [{cfg:8s}] n={n:3d} solved={ns:3d} rate={100*ns/n:5.1f}%   <paper N60: {paper}>")


merge2("Circadian", "single_task/circadian_rhythm.json",
       "single_task_actonly_72_101/circadian_rhythm.json", "95%")
merge2("Hebbian", "single_task/hebbian.json",
       "single_task_actonly_72_101/hebbian.json", "90%")
merge2("CriticalPeriod", "single_task/critical_period.json",
       "single_task_actonly_72_101/critical_period.json", "88%")
merge2("STDP", "single_task/stdp.json",
       "single_task_actonly_72_101/stdp.json", "80%")
merge2("Metaplastic", "single_task/metaplastic.json",
       "single_task_actonly_72_101/metaplastic.json", "75%")
merge2("Baseline", "single_task/baseline.json",
       "single_task_actonly_72_101/baseline.json", "65%")
merge2("Neurogenesis", "single_task/adult_neurogenesis.json",
       "single_task_actonly_72_101/adult_neurogenesis.json", "38%")

print()
print("=" * 110)
print("TABLE 3  parity scaling DUAL (parity5_single_task/, parity6_single_task/)")
print("=" * 110)
print("--- Parity-5 ---")
for s, paper in [("circadian_rhythm", "97 g30 osc100"), ("stdp", "83 g33 osc87"),
                 ("metaplastic", "83 g35 osc90"), ("clonal_selection", "83 g72 osc100"),
                 ("hebbian", "77 g34 osc83"), ("critical_period_refined", "70 g90 osc73"),
                 ("baseline", "67 g68 osc97"), ("adult_neurogenesis", "47 g37 osc87")]:
    show(s, f"parity5_single_task/{s}_dual.json", paper)
print("--- Parity-6 ---")
for s, paper in [("circadian_rhythm", "100 g24 osc80"), ("stdp", "100 g21 osc93"),
                 ("metaplastic", "100 g28 osc87"), ("clonal_selection", "100 g32 osc100"),
                 ("hebbian", "100 g33 osc77"), ("critical_period_refined", "100 g42 osc97"),
                 ("baseline", "97 g33 osc83"), ("adult_neurogenesis", "97 g44 osc100")]:
    show(s, f"parity6_single_task/{s}_dual.json", paper)

print()
print("=" * 110)
print("TABLE 4  non-parity DUAL  (non_parity/=conc,step,xor; two_moons/; visual_discrimination/)")
print("=" * 110)
print("--- Concentric Circles (paper col 1) ---")
for s, paper in [("predator_prey", "100"), ("baseline", "93"), ("clonal_selection", "90"),
                 ("adult_neurogenesis", "87"), ("metaplastic", "83"), ("stdp", "83"),
                 ("circadian_rhythm", "77"), ("critical_period_refined", "73"), ("hebbian", "53")]:
    show(s, f"non_parity/concentric_circles_{s}_dual.json", paper)
print("--- Two Moons (paper col 2) ---")
for s, paper in [("predator_prey", "100"), ("baseline", "96.7"), ("clonal_selection", "90"),
                 ("adult_neurogenesis", "96.7"), ("metaplastic", "96.7"), ("stdp", "100"),
                 ("circadian_rhythm", "83.3"), ("critical_period_refined", "100"), ("hebbian", "90")]:
    show(s, f"two_moons/{s}_dual.json", paper)
print("--- Step Function (paper col 3) ---")
for s, paper in [("predator_prey", "100"), ("baseline", "97"), ("clonal_selection", "100"),
                 ("adult_neurogenesis", "100"), ("metaplastic", "100"), ("stdp", "100"),
                 ("circadian_rhythm", "97"), ("critical_period_refined", "100"), ("hebbian", "100")]:
    show(s, f"non_parity/step_function_{s}_dual.json", paper)
print("--- Visual Discrimination (paper col 4 = all 100) ---")
for s in ["baseline", "clonal_selection", "adult_neurogenesis", "metaplastic", "stdp",
          "circadian_rhythm", "critical_period_refined", "hebbian"]:
    show(s, f"visual_discrimination/{s}_dual.json", "100")
print("  (NOTE: visual_discrimination/ has NO predator_prey file — paper lists predator-prey Vis=100)")
print("--- XOR (text: all solve 1-3 gens) ---")
for s in ["circadian_rhythm", "hebbian", "predator_prey"]:
    show(s, f"non_parity/xor_{s}_dual.json", "100 g1-3")

print()
print("=" * 110)
print("TABLE 5  tab:timescale DUAL joint (fast=single_task_dual_42_71/; slow=failure_mechanisms/)")
print("=" * 110)
for s, paper in [("hebbian", "87"), ("baseline", "73"), ("stdp", "93"),
                 ("metaplastic", "83"), ("circadian_rhythm", "90"),
                 ("critical_period_refined", "73"), ("adult_neurogenesis", "63")]:
    show(s, f"single_task_dual_42_71/{s}_dual.json", paper)
print("  --- clonal P4 dual (paper Table5 clonal=77) : searching ---")
for cand in ["single_task_n60/clonal_selection_dual.json"]:
    show("clonal(?src)", cand, "77")
print("  --- slow strategies (failure_mechanisms/) ---")
show("AntColony", "failure_mechanisms/parity_4_ant_colony_pheromone_dual.json", "80")
show("Glial", "failure_mechanisms/parity_4_glial_modulation_dual.json", "47")
show("GRN", "failure_mechanisms/parity_4_genetic_regulatory_network_dual.json", "3")

print()
print("=" * 110)
print("§6.1 / §5.6 / Oracle / Baseline-sweep / Topology / Neurogenesis-extended INLINE NUMBERS")
print("=" * 110)
print("--- GRN rescaling (DUAL) §6.1: 3%->30%, all 9 solved = band_pass+integrate / min|max agg ---")
show("GRN rescaled DUAL", "timescale_rescaling/grn_rescaled.json", "30% (9 solved)")
show("Glial rescaled DUAL", "timescale_rescaling/glial_rescaled.json", "27%")
show("Ant rescaled DUAL", "timescale_rescaling/ant_colony_rescaled.json", "-")
print("--- circadian period sensitivity §5.6: T10=67, T20=97(=Table1), T40=90 ---")
show("Circadian T10", "circadian_sensitivity/circadian_T10.json", "67% over-churn")
show("Circadian T40", "circadian_sensitivity/circadian_T40.json", "90% g25")
print("--- Oracle baseline: sin-only 100% g3.5, default+sin 93.3% g7.0, non-sin 0% ---")
show("oracle sin-only", "oracle_baseline/oracle_sin_only.json", "100% g3.5")
show("oracle sin(def+sin)", "oracle_baseline/oracle_sin.json", "93.3% g7.0")
show("baseline no_sin", "oracle_baseline/baseline_no_sin.json", "0%")
show("oracle all-18", "oracle_baseline/oracle_full_18.json", "70.0% g42")
show("oracle no-sin 17", "oracle_baseline/oracle_no_sin_17.json", "73.3% g57.5")
print("--- Oracle composite: burst 100, osc_adapt 100, resonator 63.3 ---")
show("oracle burst", "oracle_composite/oracle_burst_only.json", "100%")
show("oracle osc_adapt", "oracle_composite/oracle_osc_adapt_only.json", "100%")
show("oracle resonator", "oracle_composite/oracle_resonator_only.json", "63.3%")
print("--- Two Moons oracle: monotonic-only 93.3% ---")
show("two_moons monotonic", "oracle_two_moons/monotonic_only.json", "93.3%")
show("two_moons sin_only", "oracle_two_moons/sin_only.json", "-")
show("two_moons default+sin", "oracle_two_moons/default_plus_sin.json", "-")
print("--- Baseline sweep (mutation_rate_sweep): best-tuned 83.3%, 30%-rate median 31 ---")
for s in ["baseline_rate_05pct", "baseline_rate_10pct", "baseline_rate_15pct",
          "baseline_rate_20pct", "baseline_rate_30pct"]:
    show(s, f"mutation_rate_sweep/{s}.json", "")
print("--- Topology sensitivity single-task (full_recurrent): circadian 97->100 4x faster ---")
show("circadian FR", "topology_sensitivity_single_task/circadian_rhythm_dual_full_recurrent.json", "100%")
print("--- Neurogenesis extended: 200gen 87%, 500gen 90% ---")
show("neuro 200gen", "neurogenesis_extended/neurogenesis_200gen.json", "87%")
show("neuro 500gen", "neurogenesis_extended/neurogenesis_500gen.json", "90%")
