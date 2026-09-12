#!/usr/bin/env python3
"""Merge N=30 (seeds 42-71) and N=30 extension (seeds 72-101) into N=60 results.

D3a experiment: extend 4 key strategies from N=30 to N=60 for tighter CIs.
Strategies: baseline_dual, clonal_selection_dual, stdp_dual, circadian_rhythm_dual.
"""
import json
import os
import math
from pathlib import Path

PAPER_DIR = Path(__file__).parent
ORIGINAL_DIR = PAPER_DIR / "results" / "persistent_cl"
EXTENSION_DIR = PAPER_DIR / "results" / "persistent_cl_n60_extension"
OUTPUT_DIR = PAPER_DIR / "results" / "persistent_cl_n60_merged"

STRATEGIES = [
    "clonal_selection_dual",
    "stdp_dual",
    "circadian_rhythm_dual",
    "baseline_dual",
]

MODES = ["persistent", "reinitialized"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0, center - margin)
    hi = min(1, center + margin)
    return (round(100 * lo, 1), round(100 * hi, 1))


def extract_stage_results(runs: list, n_stages: int = 5) -> dict:
    """Extract per-stage solve counts and total stats from a list of runs."""
    n = len(runs)
    stage_solved = [0] * n_stages
    all_solved = 0
    total_stages_solved = 0

    for run in runs:
        stages = run["stages"]
        run_all = True
        for i, stage in enumerate(stages):
            if stage["solved"]:
                stage_solved[i] += 1
                total_stages_solved += 1
            else:
                run_all = False
        if run_all:
            all_solved += 1

    avg_solve_rate = round(100 * total_stages_solved / (n * n_stages), 1) if n > 0 else 0.0
    s5_rate = round(100 * stage_solved[-1] / n, 1) if n > 0 else 0.0

    return {
        "n": n,
        "stage_solved": stage_solved,
        "stage_rates": [round(100 * s / n, 1) for s in stage_solved],
        "all_solved": all_solved,
        "all_solved_rate": round(100 * all_solved / n, 1),
        "avg_solve_rate": avg_solve_rate,
        "s5_rate": s5_rate,
        "total_stages_solved": total_stages_solved,
    }


def load_runs(directory: Path, strategy: str, mode: str) -> list:
    """Load runs from a strategy+mode JSON file."""
    filename = f"{strategy}_approach_a_{mode}.json"
    filepath = directory / filename
    if not filepath.exists():
        print(f"WARNING: {filepath} not found")
        return []

    with open(filepath) as f:
        data = json.load(f)

    # Handle different JSON structures
    if "results" in data and isinstance(data["results"], dict):
        # Combined format: results -> strategy -> approach_a_mode -> runs
        key = f"approach_a_{mode}"
        for strat_key, strat_data in data["results"].items():
            if key in strat_data:
                return strat_data[key]["runs"]
    elif "results" in data and isinstance(data["results"], list):
        return data["results"]
    elif "runs" in data:
        return data["runs"]

    # Try direct access
    return data.get("runs", [])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("D3a: Merging N=30 + N=30 extension → N=60")
    print("=" * 80)

    merged_data = {}

    for strategy in STRATEGIES:
        merged_data[strategy] = {}
        print(f"\n--- {strategy} ---")

        for mode in MODES:
            orig_runs = load_runs(ORIGINAL_DIR, strategy, mode)
            ext_runs = load_runs(EXTENSION_DIR, strategy, mode)

            if not orig_runs:
                print(f"  WARNING: No original {mode} runs found")
                continue
            if not ext_runs:
                print(f"  WARNING: No extension {mode} runs found")
                continue

            # Verify no seed overlap
            orig_seeds = {r["seed"] for r in orig_runs}
            ext_seeds = {r["seed"] for r in ext_runs}
            overlap = orig_seeds & ext_seeds
            if overlap:
                print(f"  WARNING: Seed overlap: {overlap}")

            combined_runs = orig_runs + ext_runs
            n30_stats = extract_stage_results(orig_runs)
            ext_stats = extract_stage_results(ext_runs)
            n60_stats = extract_stage_results(combined_runs)

            merged_data[strategy][mode] = {
                "n30_original": n30_stats,
                "n30_extension": ext_stats,
                "n60_merged": n60_stats,
                "runs": combined_runs,
            }

            ci = wilson_ci(
                n60_stats["total_stages_solved"],
                len(combined_runs) * 5,
            )

            print(
                f"  {mode:14s}: N=30 orig={n30_stats['avg_solve_rate']:5.1f}%  "
                f"ext={ext_stats['avg_solve_rate']:5.1f}%  "
                f"N=60={n60_stats['avg_solve_rate']:5.1f}% [{ci[0]}, {ci[1]}]  "
                f"S5={n60_stats['s5_rate']:5.1f}%"
            )

    # Save merged results
    output = {
        "metadata": {
            "experiment": "D3a_n60_merged",
            "description": "Merged N=30 (seeds 42-71) + N=30 extension (seeds 72-101)",
            "strategies": STRATEGIES,
        },
        "results": {},
    }

    for strategy in STRATEGIES:
        output["results"][strategy] = {}
        for mode in MODES:
            if mode not in merged_data.get(strategy, {}):
                continue
            stats = merged_data[strategy][mode]
            output["results"][strategy][mode] = {
                "n30_original": stats["n30_original"],
                "n30_extension": stats["n30_extension"],
                "n60_merged": stats["n60_merged"],
            }

    out_path = OUTPUT_DIR / "merged_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote stats to {out_path}")

    # Print LaTeX comparison table
    print("\n" + "=" * 80)
    print("LaTeX Table Rows (N=30 vs N=60 comparison):")
    print("=" * 80)
    print(
        f"{'Strategy':<22s} {'N=30 Persist.':<16s} {'N=60 Persist.':<16s} "
        f"{'N=30 Reinit.':<16s} {'N=60 Reinit.':<16s} {'N=60 Delta':<12s}"
    )

    for strategy in STRATEGIES:
        if strategy not in merged_data:
            continue
        p_data = merged_data[strategy].get("persistent", {})
        r_data = merged_data[strategy].get("reinitialized", {})

        n30p = p_data.get("n30_original", {}).get("avg_solve_rate", 0)
        n60p = p_data.get("n60_merged", {}).get("avg_solve_rate", 0)
        n30r = r_data.get("n30_original", {}).get("avg_solve_rate", 0)
        n60r = r_data.get("n60_merged", {}).get("avg_solve_rate", 0)
        delta = n60p - n60r

        label = strategy.replace("_dual", "").replace("_", " ").title()
        print(
            f"{label:<22s} {n30p:>5.1f}%{'':9s} {n60p:>5.1f}%{'':9s} "
            f"{n30r:>5.1f}%{'':9s} {n60r:>5.1f}%{'':9s} {delta:>+5.1f}pp"
        )


if __name__ == "__main__":
    main()
