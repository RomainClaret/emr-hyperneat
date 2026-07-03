#!/usr/bin/env python3
"""A1: Mechanistic analysis of GRN-rescaled band_pass+integrate networks.

Reads solved GRN-rescaled runs and analyzes:
- Palette composition (already known: all band_pass + integrate)
- Aggregation distribution (min vs max)
- Convergence dynamics (solved_gen distribution)
- Comparison to other non-oscillatory solves (from palette_overlap analysis)

Since the JSON files only store final palette + metadata (not full network topology),
this is a statistical analysis rather than a full network reconstruction.

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/analysis/analyze_grn_mechanism.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ACTIVATION_LIST = [
    'tanh', 'sigmoid', 'relu', 'identity', 'sin', 'gauss', 'lelu',
    'softplus', 'rs_adapt', 'fs_fast', 'lts_low', 'burst', 'resonator',
    'osc_adapt', 'gain_mod', 'receptive', 'band_pass', 'integrate',
]
AGGREGATION_LIST = ['sum', 'mean', 'max', 'min', 'product', 'maxabs']
OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

RESULTS = Path(__file__).resolve().parents[2] / "results"


def analyze_grn_runs():
    """Deep dive into GRN-rescaled results across P4, P5, (P6, P7 if available)."""
    print("=" * 70)
    print("A1: GRN-RESCALED MECHANISM ANALYSIS")
    print("=" * 70)

    sources = [
        ('P4', RESULTS / 'timescale_rescaling' / 'grn_rescaled.json'),
        ('P5', RESULTS / 'timescale_rescaling_p5' / 'grn_rescaled.json'),
        ('P6', RESULTS / 'timescale_rescaling_p6p7' / 'grn_rescaled_p6.json'),
        ('P7', RESULTS / 'timescale_rescaling_p6p7' / 'grn_rescaled_p7.json'),
    ]

    for label, path in sources:
        if not path.exists():
            print(f"\n  {label}: no data")
            continue

        data = json.load(open(path))
        results = data.get('results', [])
        n = len(results)
        if n == 0:
            continue

        solved = [r for r in results if r.get('solved', False)]
        failed = [r for r in results if not r.get('solved', False)]

        print(f"\n{'─' * 60}")
        print(f"  {label}: {len(solved)}/{n} solved ({100*len(solved)/n:.1f}%)")
        print(f"{'─' * 60}")

        # Activation palette distribution in solved runs
        act_pal_counter = Counter()
        agg_pal_counter = Counter()
        osc_solved = 0
        for r in solved:
            act = tuple(sorted(r.get('final_act_palette', [])))
            agg = tuple(sorted(r.get('final_agg_palette', [])))
            act_pal_counter[act] += 1
            agg_pal_counter[agg] += 1
            if any(i in OSCILLATORY_INDICES for i in r.get('final_act_palette', [])):
                osc_solved += 1

        print(f"\n  Oscillatory presence: {osc_solved}/{len(solved)} "
              f"({100*osc_solved/len(solved) if solved else 0:.0f}%)")

        print(f"\n  Activation palettes (solved):")
        for pal, cnt in act_pal_counter.most_common():
            names = [ACTIVATION_LIST[i] for i in pal if i < len(ACTIVATION_LIST)]
            print(f"    {list(pal)} = {names}: {cnt} runs ({100*cnt/len(solved):.0f}%)")

        print(f"\n  Aggregation palettes (solved):")
        for pal, cnt in agg_pal_counter.most_common():
            names = [AGGREGATION_LIST[i] for i in pal if i < len(AGGREGATION_LIST)]
            print(f"    {list(pal)} = {names}: {cnt} runs ({100*cnt/len(solved):.0f}%)")

        # Convergence gens
        gens = [r.get('solved_gen', 0) for r in solved if r.get('solved_gen') is not None]
        if gens:
            print(f"\n  Convergence gens: median={np.median(gens):.0f}, "
                  f"mean={np.mean(gens):.1f}, range=[{min(gens)}-{max(gens)}]")

        # Failed runs: what fitness did they reach?
        if failed:
            fitnesses = [r.get('best_fitness', 0) for r in failed]
            print(f"\n  Failed runs fitness: median={np.median(fitnesses):.3f}, "
                  f"mean={np.mean(fitnesses):.3f}, range=[{min(fitnesses):.3f}-{max(fitnesses):.3f}]")

            # Near-miss analysis
            near_miss = [r for r in failed if r.get('best_fitness', 0) > 0.85]
            print(f"  Near-misses (fitness > 0.85): {len(near_miss)}/{len(failed)}")

    # Compare to other non-oscillatory solves
    print(f"\n{'=' * 70}")
    print("COMPARISON: Non-oscillatory solves across strategies")
    print(f"{'=' * 70}")
    print("""
From palette_overlap_analysis:
    hebbian: 53/219 (24%) non-oscillatory
    critical_period_refined: 39/211 (18%)
    stdp: 27/247 (11%)
    metaplastic: 21/237 (9%)
    circadian_rhythm: 20/314 (6%)
    adult_neurogenesis: 20/243 (8%)
    baseline: 11/272 (4%)
    grn: 9/9 (100%) ← ONLY 100% non-oscillatory strategy

Interpretation:
- Multiple strategies have occasional non-oscillatory solves
- These are likely concentrated on EASIER problems (P6 where osc rate is 89.5%,
  Two Moons 85.4%, Gaussian XOR 85.0%) or dual-domain runs
- GRN is uniquely 100% non-oscillatory BY DESIGN — its regulatory dynamics
  consistently lock to band_pass + integrate, while other strategies'
  non-oscillatory solves are sporadic
""")

    # Look at WHICH experiments produce non-oscillatory solves for each strategy
    print(f"{'─' * 60}")
    print("Non-oscillatory solves: which experiments contribute?")
    print(f"{'─' * 60}")

    experiments = {
        'P4_single_task': 'single_task',
        'P4_dual_domain': 'dual_domain',
        'P4_N60_extension': 'single_task_n60',
        'P5_single_task': 'parity5_single_task',
        'P6_single_task': 'parity6_single_task',
        'Gaussian_XOR': 'gaussian_xor',
        'Two_Moons': 'two_moons',
        'Visual_Discrimination': 'visual_discrimination',
    }

    for exp_name, subdir in experiments.items():
        d = RESULTS / subdir
        if not d.exists(): continue

        non_osc_by_strat = defaultdict(int)
        total_by_strat = defaultdict(int)

        for f in sorted(d.glob('*.json')):
            if f.name.startswith(('combined', 'analysis')): continue
            try:
                data = json.load(open(f))
            except Exception:
                continue
            results = data.get('results', [])
            if isinstance(results, dict): results = list(results.values())
            strat = (data.get('strategy', f.stem)).replace('_dual', '').replace('_rescaled', '')
            for r in results:
                if r.get('solved', r.get('converged', False)):
                    total_by_strat[strat] += 1
                    pal = r.get('final_act_palette', r.get('final_palette', []))
                    if isinstance(pal, list) and not any(i in OSCILLATORY_INDICES for i in pal):
                        non_osc_by_strat[strat] += 1

        # Only print experiments with at least one non-oscillatory solve
        if sum(non_osc_by_strat.values()) > 0:
            print(f"\n  {exp_name}:")
            for s in sorted(non_osc_by_strat.keys(), key=lambda x: -non_osc_by_strat[x]):
                n = non_osc_by_strat[s]
                total = total_by_strat[s]
                if n > 0:
                    print(f"    {s:<25s}: {n}/{total} non-osc ({100*n/total:.0f}%)")


if __name__ == '__main__':
    analyze_grn_runs()
