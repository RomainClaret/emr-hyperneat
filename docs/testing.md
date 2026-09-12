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
# The non-slow suite (what the nightly CI runs, spread over shards):
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -m "not slow" -q

# Everything, including the slow tests:
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -q

# A single feature file:
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_dynamic_functions.py -q

# A 5-second smoke check (base EMR on XOR), not via pytest:
JAX_PLATFORMS=cpu python emr_hyperneat/tests/test_smoke.py

# The multi-GPU strategy tests (need 2+ GPUs; they skip themselves on fewer):
JAX_PLATFORMS=gpu pytest emr_hyperneat/tests/test_multi_gpu_strategies.py -m multi_gpu -q
```

### On a GPU

The suite runs on a CUDA backend too: leave `JAX_PLATFORMS` unset. The contracts that assert one
computation across layouts (the execution modes, compaction on and off, the scan-chunked evaluation
against its loop form) are bitwise on XLA:CPU and hold to float32 rounding on a GPU, where kernels
are chosen per shape; the tests switch on `jax.default_backend()` through
`conftest.assert_trajectories_agree` (`rtol` 1e-5, set from an RTX 2080 Ti measurement whose largest
difference was 3.4e-6). Under `pytest-xdist` set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, otherwise every
worker preallocates 75% of the first card; and run `test_depth_offload_gpu.py` (marker `cuda`) on its
own with the default preallocation, because it sizes its depth from the pool XLA reports:

```bash
pytest emr_hyperneat/tests/test_depth_offload_gpu.py -m cuda -q
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest emr_hyperneat/tests -n 4 --dist loadfile -m "not cuda" -q
```

The suite is dominated by JIT compilation, not by evolution: most tests build a fresh
`EMRHyperNEAT` and compile its kernels. JAX's persistent compilation cache turns repeated
compiles of the same kernel into disk reads, within one run and across runs. Point it at a
directory and the second run of any file is several times faster:

```bash
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/emr-jax    # any writable directory
```

### On Linux, raise `vm.max_map_count` first

A full-suite run on Linux otherwise dies mid-run with `Fatal Python error: Segmentation fault`
(exit 139): XLA's CPU backend accumulates memory mappings as it compiles test after test, and a
long run crosses the kernel's per-process cap (`vm.max_map_count`, default 65530). Raise it once:

```bash
sudo sysctl -w vm.max_map_count=262144        # this boot only
echo 'vm.max_map_count=262144' | sudo tee /etc/sysctl.d/99-emr-hyperneat.conf   # persistent
```

macOS has no equivalent limit and needs nothing. Single files run fine either way — the cap only
bites accumulated full-suite runs.

## Markers

Select or deselect groups of tests with `-m`:

| Marker | Meaning |
|---|---|
| `slow` | Long-running tests. Deselect with `-m "not slow"` for the fast gate. |
| `golden_cpu` | Exact golden-value regressions. They run only on `jax==0.6.1` (install with `-c requirements-lock.txt`) and only on a CPU architecture with a recorded value (`arm64` and `x86_64`, both recorded under the canonical rule and identical to the last digit); anywhere else they skip with instructions, so the fast gate stays green on any stack. On a process whose default backend is a GPU the check runs in a CPU subprocess: the pins are CPU values. |
| `paper` | The paper-validation tests re-run a paper's simplest experiment and assert its headline finding (see below). |
| `multi_gpu` | Need 2+ GPU devices; skipped on fewer. **Deselected on GitHub** (its runners have no GPU); run locally with `-m multi_gpu`, see below. |
| `cuda` | Need one GPU with less memory than the configuration asks for. `test_depth_offload_gpu.py` is the only user: it sizes its depth from the device, so it is correct on an 11 GB card and an 80 GB one, and skips when no GPU is present. Also marked `slow`, so CI's `not slow` filter already excludes it; run it with `-m cuda`. |
| `network` | Hit live Zenodo to check the data-release fetch (`scripts/fetch_results.py`). Auto-skip when offline; run explicitly with `pytest emr_hyperneat/tests/test_data_release.py -m network`. |

> Every GPU-gated test carries both the `multi_gpu` marker and a runtime `skipif` on the device
> count, so on a laptop they report as skipped without any `-m` filter, while CI deselects them
> outright with `-m "not multi_gpu"` rather than paying to collect and skip 45 tests.

Why the goldens are keyed by architecture, and why both keys currently hold the same number: the map
is a guard, not a claim that the architectures disagree. Under the default
`variance_rule: canonical` the `arm64` and `x86_64` values are identical to the last digit. The
earlier gap was recorded under the legacy rule and appeared only in the population *mean*, which sums
50 float32 fitnesses in a backend-chosen order (the *best* is a max, order-independent, so it never
varied). Keeping the map costs nothing and keeps the 1e-9 tolerance available if a genuine
per-architecture regression ever appears.

The processor is also not what drives substrate divergence between runs that look otherwise
identical. JAX's `jax_threefry_partitionable` default flipped from `False` to `True` in JAX 0.5.0,
and it changes the initial population drawn from a given seed, which changes which regions subdivide.
Check yours with `python -c "import jax; print(jax.config.jax_threefry_partitionable)"`. Every JAX
version this package supports (`>=0.5,<0.7`) has it `True`, so substrate figures are comparable across
the supported range; comparing against numbers recorded on JAX 0.4.x needs the flag set explicitly.
The digest-level comparison behind this was run with the EMR-HyperNEAT paper's platform probes, which
are not part of this release.

Examples:

```bash
pytest emr_hyperneat/tests -m "paper and not slow"   # fast paper findings
pytest emr_hyperneat/tests -m "paper and slow"        # the slower paper findings
pytest emr_hyperneat/tests -m "not slow and not multi_gpu"   # exactly what the nightly CI runs
pytest emr_hyperneat/tests -m "not slow and not golden_cpu"  # when intentionally changing JAX
```

## What runs where

| Tier | When | What |
|---|---|---|
| Push gate (`ci.yml`, job `smoke`) | every push and pull request | the XOR smoke script, then the isolation, configuration, execution-mode, compaction, recurrence, frozen-HMR, data-release, and golden files |
| Nightly (`ci.yml`, job `full-suite`) | 04:00 UTC daily, or `workflow_dispatch` | the whole non-slow, non-GPU suite, split into shards that run in parallel; each test file is assigned to exactly one shard, and a step fails the job if a new file is not assigned |
| Paper validation (`paper-validation.yml`) | every push and pull request (fast); dispatch (slow) | the paper-validation tests below |
| Local only | on a machine with 2+ GPUs | `-m multi_gpu` (45 tests) and `-m slow` |

GitHub-hosted runners have two vCPUs and no GPU. Two things make the
suite fit them:

- **A persistent JAX compilation cache** is restored at the start of every job from
  `actions/cache` and saved at the end, keyed by OS, the runner's CPU model (XLA compiles for
  the host CPU, and the runner fleet mixes AMD and Intel), the JAX version, the job, and a hash
  of the sources, with a fallback to the latest cache of the same prefix. A run after a source
  change therefore starts warm for everything whose kernels did not change.
- **The nightly is sharded** across parallel jobs balanced by measured per-file duration, so the
  wall time is a fraction of the serial suite for the same billed minutes.

The GPU tests are the only ones GitHub cannot run at all; they are excluded by marker, not
skipped at runtime, and the local command above is the way to run them.

## The paper-validation tests

`emr_hyperneat/tests/test_paper_validation.py` is the bridge between the algorithm and the papers: it
imports the **real paper runners** and re-runs one tiny seed of each simple experiment, asserting the
published finding (e.g. an oscillatory activation solves XOR while a monotonic one does not; the
multi-task barrier and the ways around it; the EMR substrate grid scales with depth). These are
self-contained. They recompute fresh and do not need the downloaded result data.

```bash
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests/test_paper_validation.py -m "paper and not slow" -q
```

## Validating the GitHub Actions locally with `act`

CI is defined in `.github/workflows/`: `ci.yml` (the push gate and the sharded nightly) and
`paper-validation.yml` (a fast job on push/PR, a slow job on manual dispatch). Both run locally
with [`act`](https://github.com/nektos/act) (also available as the `gh act` extension) and
Docker. Pin the runner image once in `~/.actrc`:

```
-P ubuntu-latest=catthehacker/ubuntu:act-latest
```

```bash
# List jobs, then dry-run (no container execution):
act -W .github/workflows/ci.yml -l
act -W .github/workflows/ci.yml -n

# The push gate, on two cores and 7 GB like a GitHub runner (a private repo would need a token
# for the checkout; gh auth token supplies it):
act push -W .github/workflows/ci.yml -j smoke -s GITHUB_TOKEN="$(gh auth token)" \
    --container-options "--cpuset-cpus=0-1 --memory=7g"

# One nightly shard (the job's `if:` needs the dispatch event):
act workflow_dispatch -W .github/workflows/ci.yml -j full-suite --matrix shard:1 \
    -s GITHUB_TOKEN="$(gh auth token)" --container-options "--cpuset-cpus=0-1 --memory=7g"

# The paper-validation jobs:
act -W .github/workflows/paper-validation.yml -j paper-validation
act workflow_dispatch -W .github/workflows/paper-validation.yml -j paper-validation-slow
```

Notes:

- `actions/cache` works under `act` through its built-in cache server, so a second local run of
  the same job starts with a warm compilation cache, as on GitHub.
- A whole nightly shard inside a container is subject to the **host's** `vm.max_map_count`
  (see above); raise it on the host, or run the shard's files one process at a time.
- On Apple Silicon add `--container-architecture linux/amd64` to run the x86_64 runner image
  under emulation (slower, but it matches GitHub). If the image fails to pull from `ghcr.io`
  with `denied`, run `docker logout ghcr.io` and add `--pull=false` once the image is cached.

Next: [reproducing the paper experiments](reproducing-experiments.md).
