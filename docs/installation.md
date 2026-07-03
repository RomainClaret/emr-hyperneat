# Installation

EMR-HyperNEAT is a small installable package (`emr_hyperneat`) that builds on a pinned fork of
[TensorNEAT](https://github.com/RomainClaret/tensorneat), included as a git submodule. You install
both as editable packages.

## Requirements

- Python ≥ 3.10
- JAX 0.5–0.6.x (the published results and the golden-value tests were recorded on **jax 0.6.1**)
- Git (to fetch the submodule)
- ~2 GB disk for the environment (installing the pinned TensorNEAT submodule pulls heavy transitive
  dependencies, `brax`, `mujoco`, `gymnax`, even though EMR itself does not import them; on some
  platforms the `mujoco` wheel may need a system toolchain); a GPU is optional (everything runs on CPU)

## 1. Clone with the submodule

```bash
git clone --recursive https://github.com/RomainClaret/emr-hyperneat.git
cd emr-hyperneat
```

If you forgot `--recursive`:

```bash
git submodule update --init --recursive
```

The submodule lives at `third_party/tensorneat` and is pinned to a specific commit, so everyone
builds against the same NEAT/CPPN engine.

## 2. Create an environment and install

Any environment manager works (venv, conda, micromamba). Using a fresh virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
# The lockfile constraint pins the exact stack the published results were produced on
# (jax 0.6.1, flax 0.10.6, numpy 1.26.4), so newer flax/jax are not pulled in.
pip install -e . -c requirements-lock.txt                      # the emr_hyperneat package
pip install -e third_party/tensorneat -c requirements-lock.txt # the pinned TensorNEAT fork
```

> **Install, do not rely on `PYTHONPATH`.** EMR-HyperNEAT uses a `src/`-style layout for TensorNEAT
> and editable installs to resolve imports. Setting `PYTHONPATH` to the repo can silently pick up the
> wrong copy of a module. Always `pip install -e` both packages.

Optional extras:

```bash
pip install -e ".[analysis]"   # scipy + matplotlib, for the paper analysis/figure scripts
pip install -e ".[dev]"        # pytest, for the test suite
```

## 3. Pick a backend

CPU is the default and is sufficient for the algorithm, the tests, and the paper analysis scripts:

```bash
export JAX_PLATFORMS=cpu
export TF_CPP_MIN_LOG_LEVEL=3   # silence Metal/XLA init chatter on macOS
```

For a GPU, install a CUDA build of JAX and select it:

```bash
pip install "jax[cuda12]==0.6.1"
export JAX_PLATFORMS=gpu
```

Everything in this repo runs on CPU; a GPU is only needed to *re-measure* the substrate-scaling
speedups from scratch (the published tables themselves regenerate from data on CPU). Enabling the
multi-GPU path is covered in [reproducing experiments](reproducing-experiments.md#using-a-gpu).

## 4. Verify the install

```bash
JAX_PLATFORMS=cpu python emr_hyperneat/tests/test_smoke.py        # two XOR checks; ~30 s first run (JIT compile)
JAX_PLATFORMS=cpu pytest emr_hyperneat/tests -m "not slow" -q     # fast test suite
```

A clean install should print a positive fitness from the smoke test and a green pytest run. If
imports fail, double-check that **both** `pip install -e` commands succeeded and that `PYTHONPATH`
is not pointing at the repo.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: tensorneat` | You skipped `pip install -e third_party/tensorneat`, or the submodule was not checked out (`git submodule update --init --recursive`). |
| `pip` upgrades JAX to a newer version | Pin the stack exactly as above; a newer JAX changes float results and breaks the golden-value tests. |
| Metal/XLA warnings on macOS | Set `JAX_PLATFORMS=cpu` and `TF_CPP_MIN_LOG_LEVEL=3`. |
| Imports resolve to another project | Unset `PYTHONPATH`; rely on the editable installs. |

Next: [Architecture](architecture.md) for the big picture, or jump straight to
[Writing experiments](writing-experiments.md).
