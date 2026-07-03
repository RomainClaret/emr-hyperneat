# Reproducibility and Implementation Notes

Companion notes for *"Per-Node Activation Function Evolution in Indirectly Encoded
Substrates"* (ALIFE 2026): which neuroevolution implementation each experiment runs on, and
why.

## Two implementations

- **EMR-HyperNEAT**, the `emr_hyperneat` package, class `EMRHyperNEAT`. The
  framework used going forward; what this repository ships.
- **HMR-HyperNEAT**, the frozen module that produced the published results, kept under
  `emr_hyperneat/_hmr_frozen/`. Class `HMRHyperNEAT`.

EMR and HMR are **identical** in substrate discovery (byte-for-byte), RNG initialization, and
the W1/W2 **feedforward** evaluation. They produce bit-identical gen-0 populations and gen-1
fitness. They diverge only where EMR deliberately reimplemented parts of the evaluation (below).

## Which experiment runs on which

| Experiments | Module | Notes |
|---|---|---|
| **E1** parity scaling, **E2** per-function, **E7** Two Spirals, **E8** depth, **E9** lts_low | **EMR** | feedforward (global activation). Reproduce the module **bit-for-bit** with `extra_randkey_split: True` and `pop_size: 150`. |
| **E3** population sensitivity | **HMR** | `cppn_output` activation + per-node aggregation + palette evolution. |
| **E4 / E5** recurrence | **HMR** | recurrent (hidden→hidden) evaluation. |
| **E6** gradient MLP baseline |— | plain JAX MLP, no HyperNEAT. |

## Why E3/E4/E5 stay on the module

EMR reimplemented the `cppn_output` activation/aggregation and recurrence forwards, so they do
**not** bit-reproduce the module:

- **`cppn_output` activation (E3).** HMR derives each node's activation index by querying the
  CPPN activation channel at the node's **input→position connections** and averaging over
  inputs; EMR queries the node's **self-connection**. Different query position → different
  per-node functions → different fitness. (Confirmed: global-activation mode is bit-identical;
  only `cppn_output` diverges, and it diverges even with aggregation disabled.)
- **Recurrence (E4/E5).** EMR's recurrent forward (residual hidden→hidden) differs from the
  module's and consumes the PRNG stream differently.

These experiments reproduce exactly on the HMR module, which is kept for that purpose.

## The `extra_randkey_split` flag

The *only* difference between the EMR and HMR **feedforward** paths is that HMR performs one
extra `jax.random.split(state.randkey)` per generation before the ask (HMR's
`extra_randkey_split`, default `True`, which produced the paper). EMR's loop does not. The
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
full EMR-vs-HMR `_validate_emr_port.py` 18/18 check runs against the frozen HMR module.
