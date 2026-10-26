"""Record a trajectory per EMR execution mode, for comparison across modes.

History
-------
Streaming and depth offload used to be two subclasses that each forked the whole
generation loop. Folding them into one config-selected class was deliberately
behavior-changing, so "assert the modes agree" was not a valid pre-fold test:
they did not agree. This script was written to record what each mode did *before*
the fold, while the subclasses still existed, so the fold could be checked against
those recordings instead of against an assumption.

`baselines/mode_baselines.json` as committed is that pre-fold record, and it is
what the unified modes were verified against: standard, streaming and depth
offload each reproduce their recorded digest bit-for-bit. Regenerating the file
now overwrites the pre-fold record with a post-fold one: useful for comparing
modes to each other, but no longer a before/after reference. The original stays in
git history either way.

The subclass arms are gone with the subclasses; their equivalents are the
`execution.mode` settings that replaced them.

Usage:
    python capture_mode_baselines.py [--out baselines/mode_baselines.json]

Run from this directory (the test suite's conftest is imported directly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    EMRHyperNEAT,
    XORProblem,
    create_base_config,
    DEFAULT_SEED,
)

# Small on purpose: the point is a reproducible fingerprint, not a competitive run.
# Depth 3 is 340 candidate positions, which exercises multi-level subdivision while
# staying inside a few seconds per generation on CPU.
BASELINE_DEPTH = 3
BASELINE_POPULATION = 20
BASELINE_GENERATIONS = 5
BASELINE_SEED = DEFAULT_SEED

# Recorded per generation. Fitness pins the search trajectory; the substrate columns
# pin the discovered topology, which is what the variance-lookahead divergence moves.
METRIC_FIELDS = (
    "best_fitness",
    "mean_fitness",
    "min_fitness",
    "max_fitness",
    "std_fitness",
    "num_species",
    "hidden_nodes_mean",
    "connections_mean",
    "density_mean",
)


def _round(value: Any) -> Any:
    """Round floats to 10 dp so trivial float noise does not defeat comparison."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)
    return value


def _trajectory_digest(rows: List[Dict[str, Any]]) -> str:
    """Stable short hash of a whole trajectory, for at-a-glance comparison."""
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def base_config(
    depth: int = BASELINE_DEPTH,
    population: int = BASELINE_POPULATION,
    **overrides: Any,
) -> Dict[str, Any]:
    """Base config with the feature blocks pinned to their disabled state."""
    config = create_base_config(max_depth=depth, population_size=population)
    emr = config["algorithm_params"]["emrhyperneat"]["emr_hyperneat"]
    emr["recurrence"] = {"preset": "feedforward", "iteration_level": 0}
    emr["dynamic_functions"] = {"mode": "disabled"}
    emr["aggregation"] = {"mode": "disabled"}
    emr["neuromodulation"] = {"enabled": False}
    emr.update(overrides)
    # Digests are cost-independent, but keep the capture on the documented cost
    # model: reported runtimes are full-grid costs (skip_dead_levels off).
    emr.setdefault("execution", {}).setdefault("skip_dead_levels", False)
    # Record which level-0 subdivision rule produced these digests. Without this the
    # filename is the only thing distinguishing a canonical capture from a legacy one.
    emr.setdefault("variance_rule", "canonical")
    return config


def _install_branch_probe(algorithm: Any) -> Dict[str, int]:
    """Wrap the subclass generation branches so the record shows which one ran.

    Args:
        algorithm: The algorithm instance, possibly without either branch method.

    Returns:
        A live counter dict, keyed by branch name, mutated as generations run.
    """
    counts: Dict[str, int] = {}
    for attr in ("run_generation_depth_offload", "run_generation_streaming"):
        original = getattr(algorithm, attr, None)
        if original is None:
            continue
        counts[attr] = 0

        def make_counter(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
            def counted(*args: Any, **kwargs: Any) -> Any:
                counts[name] += 1
                return func(*args, **kwargs)
            return counted

        setattr(algorithm, attr, make_counter(attr, original))
    return counts


def run_arm(
    name: str,
    build: Callable[[], EMRHyperNEAT],
    config: Dict[str, Any],
    generations: int = BASELINE_GENERATIONS,
    seed: int = BASELINE_SEED,
    note: str = "",
) -> Dict[str, Any]:
    """Run one arm and record its per-generation trajectory.

    A failing arm is recorded rather than raised: an arm that cannot run today is
    itself a fact about the pre-fold code worth keeping.

    Args:
        name: Arm identifier used as the key in the baseline file.
        build: Zero-argument factory returning a configured algorithm instance.
        config: Config dict passed to `create_config`.
        generations: Number of generations to run.
        seed: RNG seed, held fixed across every arm.
        note: Human-readable statement of what this arm is meant to pin down.

    Returns:
        Record with the per-generation rows, a digest, and any error encountered.
    """
    record: Dict[str, Any] = {
        "name": name,
        "note": note,
        "seed": seed,
        "generations_requested": generations,
        "rows": [],
        "error": None,
    }
    problem = XORProblem()
    try:
        algorithm = build()
        record["class"] = type(algorithm).__name__
        config_obj = algorithm.create_config(config)
        state = algorithm.initialize(config_obj, problem, seed=seed)
    except Exception as exc:  # noqa: BLE001 - recording failure is the point
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
        return record

    # Count which branch the subclass dispatcher actually takes. Without this, an
    # arm whose threshold never fires looks identical to one that ran the offload
    # path and happened to agree with streaming. Those are opposite findings.
    branch_counts = _install_branch_probe(algorithm)

    # Positional call only: the subclasses take (state, problem, skip_metrics) and
    # raise TypeError on the parent's `verbose` keyword. That signature split is one
    # of the things the fold has to reconcile.
    for generation in range(generations):
        try:
            state, metrics = algorithm.run_generation(state, problem)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"generation {generation}: {type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=6)
            break
        record["rows"].append(
            {field: _round(getattr(metrics, field, None)) for field in METRIC_FIELDS}
        )

    record["branch_calls"] = dict(branch_counts)
    record["generations_completed"] = len(record["rows"])
    record["digest"] = _trajectory_digest(record["rows"])
    record["final_best_fitness"] = (
        record["rows"][-1]["best_fitness"] if record["rows"] else None
    )
    return record


def build_arms() -> List[Dict[str, Any]]:
    """Define every arm to capture.

    Five dispatch arms cover the branches the unified dispatcher must reproduce, and
    the divergence probes pin the specific behaviors that the fold is going to
    change, so that each change is asserted afterwards rather than assumed.
    """
    locality = 0.25  # small enough that the penalty is unmistakable if applied

    def mode_config(mode: str, **overrides: Any) -> Dict[str, Any]:
        """Base config with an explicit execution mode."""
        return base_config(execution={"mode": mode}, **overrides)

    return [
        # dispatch arms
        {
            "name": "standard",
            "build": EMRHyperNEAT,
            "config": base_config(),
            "note": "Main class, no streaming. This is the path prod's goldens lock, "
                    "and the path `mode: standard` must leave byte-identical.",
        },
        {
            "name": "flag_streaming",
            "build": EMRHyperNEAT,
            "config": base_config(enable_streaming=True, population_chunk_size=10),
            "note": "Main class with enable_streaming. Swaps only the matrix-building "
                    "step inside the normal loop, and is the tested streaming path.",
        },
        {
            "name": "depth_offload",
            "build": EMRHyperNEAT,
            "config": mode_config("depth_offload"),
            "note": "Matrices assembled in host RAM, moved to the device one chunk "
                    "at a time. Replaces the generation rather than a step in it.",
        },
        {
            "name": "auto_below_threshold",
            "build": EMRHyperNEAT,
            "config": mode_config("auto"),
            "note": "auto at a depth below both thresholds, so it must resolve to "
                    "standard and match the standard arm exactly.",
        },
        # divergence probes
        {
            "name": "probe_locality_standard",
            "build": EMRHyperNEAT,
            "config": base_config(locality_radius=locality),
            "note": "DIVERGENCE 1 reference: the main class applies locality_radius.",
        },
        {
            "name": "probe_locality_flag_streaming",
            "build": EMRHyperNEAT,
            "config": base_config(locality_radius=locality, enable_streaming=True,
                                  population_chunk_size=10),
            "note": "DIVERGENCE 1: the parent's _build_matrices_streaming does apply "
                    "locality_radius, so this should differ from flag_streaming.",
        },
        {
            "name": "probe_locality_depth_offload",
            "build": EMRHyperNEAT,
            "config": mode_config("depth_offload", locality_radius=locality),
            "note": "The offload subclass used to ignore locality_radius entirely. "
                    "After unification this must differ from depth_offload.",
        },
        {
            "name": "probe_geometry_standard",
            "build": EMRHyperNEAT,
            "config": base_config(
                geometry_seeding={"enabled": True, "seed_weight": -1.0}),
            "note": "DIVERGENCE 2 reference: geometry seeding on the main class, "
                    "which switches the CPPN to 7-D inputs.",
        },
        {
            "name": "probe_geometry_depth_offload",
            "build": EMRHyperNEAT,
            "config": mode_config(
                "depth_offload",
                geometry_seeding={"enabled": True, "seed_weight": -1.0}),
            "note": "Geometry seeding used to raise a 5-D against 7-D shape error on "
                    "the offload path. After unification it must run.",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "baselines" / "mode_baselines.json",
    )
    parser.add_argument("--generations", type=int, default=BASELINE_GENERATIONS)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated arm names, for re-running one arm.")
    args = parser.parse_args()

    arms = build_arms()
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        arms = [arm for arm in arms if arm["name"] in wanted]

    results: Dict[str, Any] = {}
    for arm in arms:
        print(f"[{arm['name']}] running ...", flush=True)
        record = run_arm(
            name=arm["name"],
            build=arm["build"],
            config=arm["config"],
            generations=args.generations,
            note=arm["note"],
        )
        results[arm["name"]] = record
        status = record["error"] or (
            f"digest={record['digest']} final={record['final_best_fitness']}"
        )
        print(f"[{arm['name']}] {status}", flush=True)

    payload = {
        "captured_on": "unified execution modes (ExecutionMode dispatch)",
        "config": {
            "problem": "XOR",
            "max_depth": BASELINE_DEPTH,
            "population_size": BASELINE_POPULATION,
            "generations": args.generations,
            "seed": BASELINE_SEED,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "arms": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")

    print("\n=== digest summary ===")
    for name, record in results.items():
        marker = "ERROR" if record["error"] else record["digest"]
        print(f"  {name:38s} {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
