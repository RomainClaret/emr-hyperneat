#!/usr/bin/env python3
"""A2: Cross-Strategy Palette Overlap Analysis.

Aggregates final_act_palette across ALL solved runs (all 8 strategies × all problems)
to identify the "palette fingerprint" of success.

Outputs:
- Frequency table: which activation indices appear in solved runs?
- Per-strategy palette composition
- Jaccard similarity matrix between strategies
- Identifies "consensus palette" (functions in >80% of solved runs)

Usage:
    python papers/emr-dynamic-functions-bio-inspired/scripts/analysis/analyze_palette_overlap.py
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
OSCILLATORY_INDICES = {4, 11, 12, 13, 15}

RESULTS_BASE = Path(__file__).resolve().parents[2] / "results"

# All experiment directories that contain palette data
EXPERIMENT_DIRS = {
    'P4_single_task': 'single_task',
    'P4_dual_domain': 'dual_domain',
    'P4_N60_extension': 'single_task_n60',
    'P5_single_task': 'parity5_single_task',
    'P6_single_task': 'parity6_single_task',
    'Gaussian_XOR': 'gaussian_xor',
    'Two_Moons': 'two_moons',
    'Visual_Discrimination': 'visual_discrimination',
    'Topology_full_recurrent': 'topology_sensitivity_single_task',
    'Timescale_rescaling': 'timescale_rescaling',
    'Pop_sensitivity_P5': 'pop_sensitivity_p5',
}


def load_solved_palettes(d: Path) -> list:
    """Return list of (strategy_name, final_palette) tuples for solved runs."""
    out = []
    if not d.exists():
        return out
    for f in sorted(d.glob('*.json')):
        if f.name.startswith(('combined', 'analysis')):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        results = data.get('results', [])
        if isinstance(results, dict):
            results = list(results.values())
        # Strategy name from filename or data
        strat = data.get('strategy', f.stem)
        # Strip variant suffixes for grouping
        for r in results:
            if r.get('solved', r.get('converged', False)):
                pal = r.get('final_act_palette', r.get('final_palette', []))
                if isinstance(pal, list) and len(pal) > 0:
                    out.append((strat, pal))
    return out


def main():
    print("=" * 70)
    print("A2: CROSS-STRATEGY PALETTE OVERLAP ANALYSIS")
    print("=" * 70)

    # Aggregate all solved palettes
    all_palettes = []  # list of (strategy, palette)
    by_experiment = defaultdict(list)

    for exp_label, subdir in EXPERIMENT_DIRS.items():
        d = RESULTS_BASE / subdir
        palettes = load_solved_palettes(d)
        all_palettes.extend(palettes)
        by_experiment[exp_label] = palettes

    print(f"\nTotal solved runs with palette data: {len(all_palettes)}")

    # === GLOBAL FREQUENCY ===
    print("\n" + "-" * 70)
    print("GLOBAL ACTIVATION FREQUENCY (across all solved runs)")
    print("-" * 70)
    func_counter = Counter()
    for _, pal in all_palettes:
        for idx in pal:
            if 0 <= idx < len(ACTIVATION_LIST):
                func_counter[idx] += 1

    total_runs = len(all_palettes)
    print(f"\n  {'Index':<6}{'Function':<15}{'In runs':<10}{'%':<8}{'Oscillatory'}")
    for idx in sorted(func_counter.keys()):
        name = ACTIVATION_LIST[idx]
        n = func_counter[idx]
        pct = 100 * n / total_runs
        osc = "OSC" if idx in OSCILLATORY_INDICES else ""
        print(f"  {idx:<6}{name:<15}{n:<10}{pct:<8.1f}{osc}")

    # === CONSENSUS PALETTE (>80% of runs) ===
    consensus = sorted([
        (idx, 100 * cnt / total_runs)
        for idx, cnt in func_counter.items()
        if cnt / total_runs > 0.8
    ], key=lambda x: -x[1])

    print(f"\n  Consensus palette (>80% of solved runs):")
    if consensus:
        for idx, pct in consensus:
            print(f"    {ACTIVATION_LIST[idx]} ({idx}): {pct:.1f}%")
    else:
        print("    None — no function appears in >80% of solved runs")

    # === PER-STRATEGY USAGE ===
    print("\n" + "-" * 70)
    print("PER-STRATEGY ACTIVATION USAGE (in their solved runs)")
    print("-" * 70)
    by_strategy = defaultdict(list)
    for strat, pal in all_palettes:
        # Normalize strategy names (strip _dual etc)
        s = strat.replace('_dual', '').replace('_rescaled', '')
        by_strategy[s].append(pal)

    print(f"\n  {'Strategy':<25}{'N':<6}{'Top 5 functions (with %)'}")
    strategy_palettes = {}
    for s, palettes in sorted(by_strategy.items()):
        n = len(palettes)
        if n == 0:
            continue
        cnt = Counter()
        for pal in palettes:
            for idx in pal:
                cnt[idx] += 1
        # Top 5
        top5 = sorted(cnt.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{ACTIVATION_LIST[i]}({100*c/n:.0f}%)" for i, c in top5 if i < len(ACTIVATION_LIST))
        print(f"  {s:<25}{n:<6}{top_str}")
        # Store for Jaccard
        strategy_palettes[s] = set(idx for pal in palettes for idx in pal)

    # === JACCARD SIMILARITY ===
    print("\n" + "-" * 70)
    print("JACCARD SIMILARITY MATRIX (between strategy palette sets)")
    print("-" * 70)
    strategies = sorted(strategy_palettes.keys())[:10]  # top 10 to keep readable
    if len(strategies) >= 2:
        print(f"\n  {'':25s}", end='')
        for s2 in strategies:
            print(f"{s2[:8]:>10s}", end='')
        print()
        for s1 in strategies:
            print(f"  {s1:<25s}", end='')
            for s2 in strategies:
                set1 = strategy_palettes[s1]
                set2 = strategy_palettes[s2]
                if len(set1 | set2) == 0:
                    j = 0
                else:
                    j = len(set1 & set2) / len(set1 | set2)
                print(f"{j:>10.2f}", end='')
            print()

    # === OSCILLATORY VS NON-OSCILLATORY SOLVES ===
    print("\n" + "-" * 70)
    print("OSCILLATORY VS NON-OSCILLATORY SOLVES")
    print("-" * 70)
    osc_solved = sum(1 for _, pal in all_palettes if any(i in OSCILLATORY_INDICES for i in pal))
    non_osc_solved = total_runs - osc_solved
    print(f"\n  Oscillatory solved:     {osc_solved}/{total_runs} ({100*osc_solved/total_runs:.1f}%)")
    print(f"  Non-oscillatory solved: {non_osc_solved}/{total_runs} ({100*non_osc_solved/total_runs:.1f}%)")

    # Which strategies solved without oscillatory?
    print("\n  Non-oscillatory solves by strategy:")
    no_osc_by_strat = defaultdict(int)
    total_by_strat = defaultdict(int)
    for strat, pal in all_palettes:
        s = strat.replace('_dual', '').replace('_rescaled', '')
        total_by_strat[s] += 1
        if not any(i in OSCILLATORY_INDICES for i in pal):
            no_osc_by_strat[s] += 1
    for s in sorted(no_osc_by_strat.keys(), key=lambda x: -no_osc_by_strat[x]):
        if no_osc_by_strat[s] > 0:
            print(f"    {s}: {no_osc_by_strat[s]}/{total_by_strat[s]} ({100*no_osc_by_strat[s]/total_by_strat[s]:.0f}%)")

    # === EXPERIMENT BREAKDOWN ===
    print("\n" + "-" * 70)
    print("PER-EXPERIMENT OSCILLATORY STATS")
    print("-" * 70)
    print(f"\n  {'Experiment':<35}{'Solved':<10}{'Osc.':<10}{'%'}")
    for exp_label in sorted(EXPERIMENT_DIRS.keys()):
        palettes = by_experiment[exp_label]
        if not palettes:
            continue
        n = len(palettes)
        osc = sum(1 for _, p in palettes if any(i in OSCILLATORY_INDICES for i in p))
        print(f"  {exp_label:<35}{n:<10}{osc:<10}{100*osc/n:.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print(f"  Total solved runs analyzed: {total_runs}")
    print(f"  Oscillatory presence rate: {100*osc_solved/total_runs:.1f}%")
    if consensus:
        print(f"  Consensus palette: {[ACTIVATION_LIST[i] for i, _ in consensus]}")
    else:
        print(f"  No global consensus — strategies use diverse palettes")
    print()


if __name__ == '__main__':
    main()
