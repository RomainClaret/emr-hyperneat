# Testing

The repository ships a test suite (over 1,900 tests) covering every published feature plus
isolation, determinism, and EMR-vs-HMR equivalence guards (against
[HMR's recorded values](#hmrs-recorded-values)). Run it from the installed
environment (see [installation](installation.md)); do not rely on `PYTHONPATH`.

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
`conftest.assert_trajectories_agree` (`rtol` 1e-4, set from an RTX 2080 Ti measurement whose largest
difference was 3.4e-6). The CPU contracts hold bitwise on the arm64 and x86 processors the suite ran
on, with one exception on x86: bucketed compaction can round a padded sum differently from `exact`,
so the comparison of the standard and depth-offload modes runs both at exact shapes. The recurrent
substrate API's forwards are compared on a GPU in the genomes that one ulp in the weights does not
move past that tolerance (`test_substrate_recurrent.py`): with dynamic h->h aggregation a GPU's
forward differs from itself between calls. Under
`pytest-xdist` set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, otherwise every
worker preallocates 75% of the first card; and run `test_depth_offload_gpu.py` (marker `cuda`) on its
own with the default preallocation and allocator, because it sizes its depth from the pool XLA
reports (`XLA_PYTHON_CLIENT_ALLOCATOR=platform` reports none, and its tests skip):

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

macOS has no equivalent limit and needs nothing. Single files run fine either way: the cap only
bites accumulated full-suite runs.

## Markers

Select or deselect groups of tests with `-m`:

| Marker | Meaning |
|---|---|
| `slow` | Long-running tests. Deselect with `-m "not slow"` for the fast gate. |
| `golden_cpu` | Exact golden-value regressions. They run only on `jax==0.6.1` (install with `-c requirements-lock.txt`) and only on a CPU architecture with a recorded value (`arm64` and `x86_64`, both recorded under the canonical rule and identical to the last digit); anywhere else they skip with instructions, so the fast gate stays green on any stack. On a process whose default backend is a GPU the check runs in a CPU subprocess: the pins are CPU values. |
| `paper` | The paper-validation tests re-run a paper's simplest experiment and assert its headline finding (see below). |
| `multi_gpu` | Need GPU devices and skip without them: all 26 need at least two (one of them four). **Deselected on GitHub** (its runners have no GPU); run locally with `-m multi_gpu`, see below. |
| `cuda` | Need one GPU with less memory than the configuration asks for. `test_depth_offload_gpu.py` is the only user: it sizes its depth from the device, so it is correct on an 11 GB card and an 80 GB one, and skips when no GPU is present. Also marked `slow`, so CI's `not slow` filter already excludes it; run it with `-m cuda`. |
| `network` | Hit live Zenodo to check the data-release fetch (`scripts/fetch_results.py`). Auto-skip when offline; run explicitly with `pytest emr_hyperneat/tests/test_data_release.py -m network`. |

> Every multi-GPU test carries the `multi_gpu` marker and skips when the GPUs it needs are absent:
> all 26 through a runtime `skipif` on the device count.
> So on a laptop they report as skipped without any `-m` filter, while CI deselects them with
> `-m "not multi_gpu"`, which acts after collection: it saves the 26 skips, not the collection.

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

## HMR's recorded values

Some tests compare EMR against HMR, the engine of the published results that EMR replaced:
`test_hmr_reference.py` (EMR reproduces HMR's dynamic-functions variant, every generation's best
fitness and the final random key) and the HMR-compatibility tests of `test_recurrent_rule.py`,
`test_hmr_names.py`, `test_unified_recipe.py`, `test_live_palette_strategy.py`,
`test_anti_node_rule.py`, `test_leo_hmr_compound.py` and `test_additive_settling.py`, and the
comparisons of `test_node_list.py` with HMR's recurrent-network variant and its GPU-resident variant
(the networks, fitness and populations of their runs, their cleaning and forward). In the research
repository, where HMR lives, these tests run it. Here they do not run HMR: they read
HMR's side from `emr_hyperneat/tests/hmr_reference_values.py` through `hmr_reference.hmr_value`.
The values were recorded in the research repository, on the CPU, and only after every one of those
tests had passed there against HMR itself. Each value is kept with the sha256 of the inputs it was
computed from (configurations, problems, seeds and, where HMR ran on EMR's networks, the arrays EMR
built), the CPU architecture, the JAX version and the research commit. Each architecture was
recorded twice, with `XLA_FLAGS` unset and under `--xla_cpu_multi_thread_eigen=false` (the nightly
job's), and the two recordings are merged: on x86-64, single-threaded Eigen rounds some of EMR's
arithmetic differently, so a comparison whose inputs include arrays EMR computed hashes differently
under each. Under other `XLA_FLAGS` such a test can miss its recorded inputs; its failure then names
the conditions the values were recorded under.

A recorded value is weaker than a live check, in three ways:

- It holds only for the inputs it was recorded for. When EMR's side of a comparison changes, the
  test fails on the inputs' hash instead of comparing, and only a new recording against HMR, in the
  research repository, can tell whether EMR still matches HMR.
- It holds only where it was recorded. Values are kept per CPU architecture (`arm64`, `x86_64`, and
  each case says whether the two agree); on an architecture with no record, on another JAX version,
  or off the CPU, the test skips instead of checking. On a GPU, `test_hmr_reference.py` runs itself
  in a CPU subprocess, as the goldens do.
- It shows that EMR computes what HMR computed when the value was recorded. The tests cannot run
  HMR again, so they cannot catch an error in the recording itself; the recording's own check,
  every test passing against HMR before anything is written, is the guard.

## What runs where

| Tier | When | What |
|---|---|---|
| Push gate (`ci.yml`, job `smoke`) | every push and pull request | the XOR smoke script, then the isolation, configuration, execution-mode, compaction, recurrence, recorded-HMR, data-release, golden, text-style, multi-GPU worker, paper-import and pytest-configuration files |
| Nightly (`ci.yml`, job `full-suite`) | 04:00 UTC daily, or `workflow_dispatch` | the whole non-slow, non-GPU suite, split into shards that run in parallel; each test file is assigned to exactly one shard, and a step fails the job if a new file is not assigned. It includes `test_multi_device_parity.py`, which runs every multi-device route on two, three and four forced CPU devices and checks that each hands NEAT the generation-0 fitness and draws the random keys a single-device run does. The longer harnesses on forced CPU devices are marked `slow`: the split routes' runs and backends against `single_gpu`, bit for bit (`test_multi_device_parity.py`, `test_device_split.py`), and the island placements (`test_islands.py`) |
| Paper validation (`paper-validation.yml`) | every push and pull request (fast); dispatch (slow, and `replicate`) | the paper-validation tests below; on dispatch, also each paper's own release, through `scripts/replicate.py` |
| Local only | on a machine with 2+ GPUs | `-m multi_gpu` (26 tests) and `-m slow` |

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

Their golden values were recorded on each paper's own release, so they check that the current
release still gives a paper's results. The exception is the bio-inspired paper's golden
(`TestBioInspiredPaper`): its runner ran the HMR engine in 0.1.0 and is an EMR version from 0.2.0,
where the golden was recorded again. The `replicate` job (manual dispatch) checks the other
direction: it builds every release that `papers/releases.json` names with `scripts/replicate.py`
and runs that release's own fast paper findings in its checkout
([choosing a release](reproducing-experiments.md#choosing-a-release)), then release 0.1.0's
`TestBioInspiredPaper` and `TestAggregationOraclePaper`, whose runners ran the HMR engine there. `test_replicate.py` checks
the script without network or installs: the release map against `papers/` and the READMEs, and
its git steps in a temporary repository with a tag.

## Validating the GitHub Actions locally with `act`

CI is defined in `.github/workflows/`: `ci.yml` (the push gate and the sharded nightly) and
`paper-validation.yml` (a fast job on push/PR; a slow job and `replicate` on manual dispatch).
Both run locally with [`act`](https://github.com/nektos/act) (also available as the `gh act`
extension) and Docker. Pin the runner image once in `~/.actrc`:

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
act workflow_dispatch -W .github/workflows/paper-validation.yml -j replicate  # needs the tags
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
