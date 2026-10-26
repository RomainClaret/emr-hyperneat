# Reproducibility and Implementation Notes

Companion notes for *"Per-Node Activation Function Evolution in Indirectly Encoded
Substrates"* (ALIFE 2026): which neuroevolution implementation each experiment runs on, and
why.

## Two implementations

- **EMR-HyperNEAT**, the `emr_hyperneat` package, class `EMRHyperNEAT`. The
  framework this repository ships.
- **HMR-HyperNEAT**, the engine that produced the published results. Release 0.1.0 ships
  it as a frozen module, `emr_hyperneat/_hmr_frozen/` (class `HMRHyperNEAT`); 0.2.0 does not.

With EMR pinned to `variance_rule: legacy` (the package default is `canonical`),
EMR and HMR are **identical** in substrate discovery (byte-for-byte), RNG initialization, and
the W1/W2 **feedforward** evaluation. They produce bit-identical gen-0 populations and gen-1
fitness. They diverge only where EMR deliberately reimplemented parts of the evaluation (below).

## Which experiment runs on which

| Experiments | Module | Notes |
|---|---|---|
| **E1** parity scaling, **E2** per-function, **E7** Two Spirals, **E8** depth, **E9** lts_low | **EMR** | feedforward (global activation). Reproduce the module **bit-for-bit** with `extra_randkey_split: True`, `pop_size: 150`, and `variance_rule: legacy`. |
| **E3** population sensitivity | **HMR** (published); **EMR** from 0.2.0 | `cppn_output` activation + per-node aggregation + palette evolution. |
| **E4 / E5** recurrence | **HMR** (published); **EMR** from 0.2.0 | recurrent (hidden->hidden) evaluation. |
| **E6** gradient MLP baseline | none | plain JAX MLP, no HyperNEAT. |

## E3/E4/E5 from release 0.2.0

EMR reimplemented the `cppn_output` activation/aggregation and recurrence forwards, so they do
**not** bit-reproduce the HMR engine:

- **`cppn_output` activation (E3).** HMR derives each node's activation index by querying the
  CPPN activation channel at the node's **input->position connections** and averaging over
  inputs; EMR queries the node's **self-connection** (its default; `cppn_read: input_mean` is
  HMR's read). Different query position -> different per-node functions -> different fitness.
  (Confirmed: global-activation mode is bit-identical; only `cppn_output` diverges, and it
  diverges even with aggregation disabled.)
- **Recurrence (E4/E5).** HMR settled by `h = f(x W1)` once, then `h = tanh(h + recurrent
  input)`, kept the hidden-to-hidden connection cache on and composed its multi-hop expansion
  through the population's union of active positions. EMR settles by `h = f(input drive +
  recurrent input)` at every step, with the cache off and each genome's own connections (its
  defaults).

From 0.2.0 the three runners are EMR versions that run each experiment as it ran, on EMR's
defaults: population 150 for E3 (the published arms labeled 200 and 1000 all ran at 150; the
argument is kept and ignored) and 500 for E4 and E5, the recurrent presets at iteration level 2
(the published level; the preset default is 1 from 0.2.0), with HMR's level-0 variance rule and
random key split, which `scripts/runners/emr_settings.py` pins as the feedforward runners pin them
(set to False, it runs EMR's defaults). Each runner's docstring
says what differs, and each writes apart from the published runs, to `results/<experiment>_emr/`
(E3 to the bio-inspired paper's `results/population_sensitivity_emr/`, as its published runs went
to that paper's folder). The published runs reproduce on release 0.1.0, which ships the HMR engine:

```bash
python scripts/replicate.py run emr-dynamic-functions --release 0.1.0 -- \
    python scripts/runners/E5_recurrence_type.py --preset hidden_only
```

## The `extra_randkey_split` flag

Two things differ between the EMR and HMR **feedforward** paths, and both are pinned by the
runners. First, HMR performs one extra `jax.random.split(state.randkey)` per generation before
the ask (HMR's `extra_randkey_split`, default `True`, which produced the paper); EMR's loop does
not. Second, EMR's level-0 subdivision variance defaults to the canonical
per-parent rule, while the frozen HMR module hard-wires the legacy rule and takes no flag, so
EMR must be pinned with `variance_rule: legacy`. The
opt-in EMR config key `emr_hyperneat.extra_randkey_split` (default **False**, so EMR's normal
behavior is unchanged) restores that split, yielding bit-identical per-seed trajectories. The
five feedforward runners set it to `True`.

`pop_size = 150`: HMR's population wiring fixed the population at 150 regardless of the nominal
value the runner passed; EMR honors the config, so the ported runners pin 150 to match the
published results.

## Verify

```bash
# E1/E2/E7/E8/E9 (EMR path): run any runner, e.g.
JAX_PLATFORMS=cpu python scripts/runners/E2_xor_per_function.py --function sin --seeds 2 --max-gens 10
```

The EMR feedforward runners bit-reproduce the frozen HMR module (verified: seed 42 -> solved
gen 1, fit 0.999983; seed 43 -> gen 1, 0.999992, matching the published per-seed values). The
full EMR-vs-HMR check, `python scripts/analysis/_validate_emr_port.py`, compares EMR with HMR's
trajectories of its 18 cells, recorded on release 0.1.0 by that release's own version of the
script and kept per CPU architecture in `scripts/analysis/hmr_trajectories.json` (with the command
that recorded them). It reports 18/18 cells bit-identical, every generation's best fitness and the
final random key, on arm64 and x86_64 with XLA_FLAGS unset and under
`--xla_cpu_multi_thread_eigen=false` (it pins `variance_rule: legacy` and `extra_randkey_split` for
the reasons above), and writes its comparison to `results/validate_emr_port_emr/validation.json`.
