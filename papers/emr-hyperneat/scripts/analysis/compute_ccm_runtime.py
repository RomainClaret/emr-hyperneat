#!/usr/bin/env python3
"""CCM financial-dataset runtime: ES-HN vs EMR total runtime over 30 generations.

Reads the real 30-generation runs in ``data/runs_ccm.json`` (CRSP/Compustat Merged data,
94,464 samples, pop=100), with one ES-HN and one EMR run per depth (1-6). ES-HyperNEAT
(PUREPLES, CPU) has no JIT, so its construction is not folded into generation 0
(``construction_in_gen0`` is False); the EMR run (GPU) pays JIT as generation 0
(``construction_in_gen0`` is True). That flag labels the two runs.

Output: per-depth ES-HN / EMR totals, the crossover speedup, and TikZ coordinates.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CCM_JSON = DATA_DIR / "runs_ccm.json"
N_GENS = 30


def format_time(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}hr"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.1f}s"


def load_ccm():
    """Return {depth: {'es_hn': total_s, 'emr': total_s}} from the real runs.

    The CPU ES-HN run has no JIT (construction_in_gen0 == False); the EMR run of the
    pair pays JIT as generation 0 (construction_in_gen0 == True).
    """
    runs = json.load(open(CCM_JSON))["runs"]
    by_depth = {}
    for r in runs:
        key = "emr" if r["construction_in_gen0"] else "es_hn"
        by_depth.setdefault(r["depth"], {})[key] = r["total_runtime_s"]
    for d, pair in by_depth.items():
        if set(pair) != {"es_hn", "emr"}:
            raise ValueError(f"Depth {d}: expected one ES-HN and one EMR run, got {pair}")
    return dict(sorted(by_depth.items()))


def main():
    by_depth = load_ccm()

    print(f"CCM Runtime Analysis ({N_GENS} generations, real runs)")
    print("=" * 70)
    print(f"\n{'Depth':<6} {'ES-HN':<22} {'EMR':<22} {'Speedup':<10}")
    print("-" * 60)
    for d, pair in by_depth.items():
        es, emr = pair["es_hn"], pair["emr"]
        speedup = es / emr
        print(f"{d:<6} {es:>8.1f}s ({format_time(es):<7}) "
              f"{emr:>8.1f}s ({format_time(emr):<7}) {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("LaTeX plot coordinates:")
    print("-" * 70)
    es_coords = " ".join(f"({d}, {pair['es_hn']:.1f})" for d, pair in by_depth.items())
    emr_coords = " ".join(f"({d}, {pair['emr']:.1f})" for d, pair in by_depth.items())
    print(f"CCM ES-HN: {es_coords}")
    print(f"CCM EMR:   {emr_coords}")

    # Crossover: the deepest depth where EMR overtakes ES-HN.
    crossover = [d for d, p in by_depth.items() if p["emr"] < p["es_hn"]]
    if crossover:
        d = max(crossover)
        es, emr = by_depth[d]["es_hn"], by_depth[d]["emr"]
        print(f"\nCaption: at depth {d}, EMR achieves {es / emr:.1f}x speedup "
              f"({emr:.0f}s vs {es:.0f}s)")


if __name__ == "__main__":
    main()
