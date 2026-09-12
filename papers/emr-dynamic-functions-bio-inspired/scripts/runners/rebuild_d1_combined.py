#!/usr/bin/env python3
"""Rebuild D1 combined_results.json from individual strategy JSON files.

The original combined_results.json only contained sin_default_dual (the last strategy run).
This script rebuilds it from all 7 individual strategy files.
"""
import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "reverse_cross_problem_cl"

STRATEGIES = [
    "baseline_dual",
    "clonal_selection_dual",
    "circadian_rhythm_dual",
    "stdp_dual",
    "consolidation_window_dual",
    "sin_default_dual",
    "critical_period_refined_dual",
]


def compute_summary(strategy_name: str, results: list) -> dict:
    """Compute summary statistics for a strategy."""
    n = len(results)
    s1_solved = sum(1 for r in results if r["stages"][0]["solved"])
    s2_solved = sum(1 for r in results if r["stages"][1]["solved"])
    s3_solved = sum(1 for r in results if r["stages"][2]["solved"])
    all_3 = sum(1 for r in results if all(s["solved"] for s in r["stages"]))

    osc_retained = sum(
        1 for r in results
        if r["stages"][2].get("has_oscillatory", False)
    )

    return {
        "strategy": strategy_name,
        "n": n,
        "stage1_solve_rate": round(100 * s1_solved / n, 1),
        "stage2_solve_rate": round(100 * s2_solved / n, 1),
        "stage3_solve_rate": round(100 * s3_solved / n, 1),
        "all_3_solve_rate": round(100 * all_3 / n, 1),
        "osc_retained_rate": round(100 * osc_retained / n, 1),
    }


def main():
    combined = {
        "metadata": {
            "experiment": "D1_reverse_cross_problem_cl",
            "strategies": STRATEGIES,
            "sequence": "concentric_circles -> parity_4 -> concentric_circles",
            "description": "Reverse cross-problem CL: CC(50) -> P4(50) -> CC(50)",
            "total_runs": 0,
            "total_runtime_seconds": 0.0,
        },
        "results": {},
    }

    total_runs = 0
    total_time = 0.0

    for strategy in STRATEGIES:
        filepath = RESULTS_DIR / f"{strategy}.json"
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        results = data["results"]
        n = len(results)
        total_runs += n

        # Sum runtime from individual stages
        strategy_time = 0.0
        for run in results:
            for stage in run["stages"]:
                strategy_time += stage.get("time_seconds", 0.0)
        total_time += strategy_time

        summary = compute_summary(strategy, results)

        combined["results"][strategy] = {
            "strategy": strategy,
            "sequence": data.get("sequence", "CC -> Parity-4 -> CC"),
            "summary": summary,
            "results": results,
        }

        print(
            f"{strategy:35s}: S1={summary['stage1_solve_rate']:5.1f}%  "
            f"S2={summary['stage2_solve_rate']:5.1f}%  "
            f"S3={summary['stage3_solve_rate']:5.1f}%  "
            f"All3={summary['all_3_solve_rate']:5.1f}%  "
            f"Osc.Ret={summary['osc_retained_rate']:5.1f}%  (n={n})"
        )

    combined["metadata"]["total_runs"] = total_runs
    combined["metadata"]["total_runtime_seconds"] = total_time
    combined["metadata"]["seeds"] = list(range(42, 72))

    out_path = RESULTS_DIR / "combined_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nWrote {out_path} with {len(combined['results'])} strategies, {total_runs} total runs")


if __name__ == "__main__":
    main()
