# Testing

The repository ships a test suite (900+ tests) covering every published feature plus isolation,
determinism, and EMR-vs-frozen-HMR equivalence guards. Run it from the installed environment (see
[installation](installation.md)); do not rely on `PYTHONPATH`.

Install the test dependency once (pytest is in the `dev` extra, not the base install):

```bash
pip install -e ".[dev]"
```

## Run the suite

```bash
# Fast gate, what CI runs on every push (~minutes on CPU):
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -m "not slow" -q

# Everything, including the slow tests:
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -q

# A single feature file:
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_dynamic_functions.py -q

# A 5-second smoke check (base EMR on XOR), not via pytest:
JAX_PLATFORMS=cpu python emr_hyperneat/tests/test_smoke.py
```

## Markers

Select or deselect groups of tests with `-m`:

| Marker | Meaning |
|---|---|
| `slow` | Long-running tests. Deselect with `-m "not slow"` for the fast gate. |
| `golden_cpu` | Exact golden-value regressions recorded on CPU with `jax==0.6.1`. They auto-skip on any other JAX version (the goldens are stack-specific), so the fast gate stays green regardless of your JAX; install with `-c requirements-lock.txt` to run them. |
| `paper` | The paper-validation tests re-run a paper's simplest experiment and assert its headline finding (see below). |
| `multi_gpu` | Require 2+ GPUs; skipped on a single device. |
| `cuda` | Require a CUDA build of JAX. |
| `network` | Hit live Zenodo to check the data-release fetch (`scripts/fetch_results.py`). Auto-skip when offline; run explicitly with `pytest emr_hyperneat/tests/test_data_release.py -m network`. |

> GPU tests (multi-GPU strategies and CUDA paths) detect the available devices at runtime and **skip
> themselves** on a CPU-only machine. No `-m` filter is needed; they simply report as skipped.

Examples:

```bash
pytest emr_hyperneat/tests -m "paper and not slow"   # fast paper findings
pytest emr_hyperneat/tests -m "paper and slow"        # the slower paper findings
pytest emr_hyperneat/tests -m "not slow and not golden_cpu"
```

## The paper-validation tests

`emr_hyperneat/tests/test_paper_validation.py` is the bridge between the algorithm and the papers: it
imports the **real paper runners** and re-runs one tiny seed of each simple experiment, asserting the
published finding (e.g. an oscillatory activation solves XOR while a monotonic one does not; the
multi-task barrier and the ways around it; the EMR substrate grid scales with depth). These are
self-contained. They recompute fresh and do not need the downloaded result data.

```bash
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_paper_validation.py -m "paper and not slow" -q
```

## Validating the GitHub Action locally with `act`

CI is defined in `.github/workflows/`. The `paper-validation.yml` workflow has two jobs: a fast job
(runs on push/PR) and a slow job (manual dispatch only). You can run them locally with
[`act`](https://github.com/nektos/act) and Docker.

```bash
# Lint + list + dry-run (fast, no container execution):
yamllint -d relaxed .github/workflows/paper-validation.yml
act -W .github/workflows/paper-validation.yml --list
act -W .github/workflows/paper-validation.yml -n

# Run the fast job for real in a container:
act -W .github/workflows/paper-validation.yml -j paper-validation \
    --container-architecture linux/amd64

# Run the manual slow job, note the workflow_dispatch event is required,
# otherwise the job's `if:` gate skips it:
act workflow_dispatch -W .github/workflows/paper-validation.yml -j paper-validation-slow \
    --container-architecture linux/amd64
```

Notes for Apple-Silicon machines:

- `--container-architecture linux/amd64` runs the standard x86_64 runner image under emulation. It is
  slower but matches GitHub's runners.
- If the runner image fails to pull from `ghcr.io` with `denied`, run `docker logout ghcr.io` (a
  stale credential forces an authenticated pull) and add `--pull=false` once the image is cached.

Next: [reproducing the paper experiments](reproducing-experiments.md).
