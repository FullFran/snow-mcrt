# GPU validation results

Produced by `kaggle/run.py` on a Kaggle GPU kernel and committed here so the
claims in the documentation can be checked without anyone re-running them.

**Run:** https://www.kaggle.com/code/fran17/snow-mcrt-gpu-validation
**Hardware:** Tesla P100-PCIE-16GB, 15.89 GB, CuPy 14.0.1
**Date:** 2026-07-28

## What it establishes

**The CuPy adapter works.** Until this run it had never executed numerically —
without CUDA its four numerical tests skip, leaving only the contract tests. An
adapter that has never run is an intention, not an adapter.

Against the NumPy oracle, semi-infinite isotropic transport:

| `omega` | NumPy | CuPy | relative difference | van de Hulst |
| ------- | ----- | ---- | ------------------- | ------------ |
| 0.50 | 0.14694 | 0.14620 | 0.50% | 0.14453 |
| 0.90 | 0.47881 | 0.47856 | 0.053% | 0.47717 |
| 0.95 | 0.59751 | 0.59741 | 0.016% | 0.59627 |

The two backends use different random number generators, so bit-identical
results are not expected — statistical agreement is. It tightens as `omega`
rises, which is the right direction: more photons survive longer and the
estimator has more to work with.

## The GPU is not always faster, and the numbers say so

| `omega` | NumPy | CuPy | speedup |
| ------- | ----- | ---- | ------- |
| 0.50 | 0.31 s | 3.68 s | **0.08x** |
| 0.90 | 1.26 s | 0.21 s | 6.0x |
| 0.95 | 2.94 s | 0.36 s | 8.2x |

At `omega = 0.5` the transport finishes in about two dozen scattering orders
and kernel launch overhead dominates completely — the GPU is twelve times
*slower*. This is worth stating plainly because it is the same conclusion the
architecture notes reached from the other direction: the GPU pays in deep
transport, not in anything that merely fits in an array.

These timings are from the run that also produced the diffusion section below,
and they replace an earlier run's — 0.07x, 6.5x, 7.4x. The spread between the
two is up to 20%, which is what a shared Kaggle GPU does to a wall clock.

The *physics* did not move at all. Every Monte Carlo value in the next table
is bit-identical across the two runs, because the seed is part of the
configuration and the transport is deterministic given it. Timing is the one
thing here that is not reproducible, and separating the two is the reason both
are reported.

## Deep transport, 500 000 photons

| wavelength | `1 - omega` | Monte Carlo | analytic | difference | scattering orders | time | truncated |
| ---------- | ----------- | ----------- | -------- | ---------- | ----------------- | ---- | --------- |
| 700 nm | 6.5e-05 | 0.94517 | 0.94497 | +0.021% | 50 548 | 83.8 s | 0 |
| 900 nm | 7.3e-04 | 0.82808 | 0.82746 | +0.075% | 5 118 | 8.4 s | 0 |
| 1100 nm | 2.4e-03 | 0.71046 | 0.70947 | +0.139% | 1 543 | 2.6 s | 0 |

`truncated = 0` throughout: every photon either escaped or was absorbed within
budget. Nothing was swept into the ledger to make the albedo look converged.

Note the cost scaling. Dropping `1 - omega` by a factor of 37 (1100 nm to
700 nm) raises the scattering orders by a factor of 33 and the wall clock by a
factor of 32. That is the `1/(1 - omega)` law, measured.

## What is still not reached

Clean snow at 500 nm sits at `1 - omega ~ 5e-6`, another factor of thirteen
below the hardest case here, and would need of order a million scattering
orders. At 500 000 photons that is over an hour of wall clock even on this
hardware. It is a benchmark of patience rather than of physics, so it is not
run. The engine reports `truncated` precisely so that if anyone does run it
under-budgeted, the shortfall is visible rather than absorbed into a
plausible-looking albedo.

## Where diffusion holds — the 3-D engine against the closed form

Added with the 3-D transport engine. Every figure in
[`docs/detectability.md`](../../docs/detectability.md) is a diffusion
calculation, and none had been checked against transport. These runs measure
the approximation: **two million photons per case**, 400 000 scattering orders,
four snowpacks spanning three orders of magnitude in co-albedo.

| case | `1 - omega` | `delta` | reflected | 3–12 `mfp'` | all `< 12 mfp'` | truncated |
| ---- | ----------- | ------- | --------- | ----------- | --------------- | --------- |
| clean, 450 nm | 3.3e-07 | 91.3 cm | 0.9915 | **9.0%** | 19.9% | 3.0e-03 |
| Arctic, 450 nm | 7.9e-06 | 18.5 cm | 0.9633 | **9.3%** | 20.1% | 1.5e-09 |
| alpine, 450 nm | 7.6e-05 | 6.0 cm | 0.8912 | **10.6%** | 20.5% | 0 |
| alpine, 800 nm | 3.0e-04 | 3.0 cm | 0.7950 | **10.9%** | 21.0% | 0 |

Diffusion is good to about 10% over the separations a source-detector pair
uses. The `< 12 mfp'` column is twice that because it includes the near field,
where diffusion is not inaccurate but *inapplicable* — the innermost bin is
0.79–0.80 in every case. Quote the first column.

Per-bin profiles are in `mc-diffusion-*.csv`.

### Why this needed a GPU, with the receipt

Not throughput — convergence. Clean snow at 450 nm sits at `1 - omega = 3.3e-07`,
so a photon needs of order a million scattering orders before it is absorbed,
and a truncated run does not fail, it quietly returns a low reflectance.

The same four cases at 120 000 photons and 6 000 orders on one CPU core:

| case | reflected, CPU | reflected, GPU | difference | truncated, CPU | truncated, GPU |
| ---- | -------------- | -------------- | ---------- | -------------- | -------------- |
| clean, 450 nm | 0.9548 | 0.9915 | **+0.0367** | 4.4e-02 | 3.0e-03 |
| Arctic, 450 nm | 0.9463 | 0.9633 | +0.0170 | 3.6e-02 | 1.5e-09 |
| alpine, 450 nm | 0.8905 | 0.8912 | +0.0007 | 5.1e-03 | 0 |
| alpine, 800 nm | 0.7954 | 0.7950 | -0.0004 | 6.7e-06 | 0 |

The CPU run is wrong by 3.7 points of albedo on clean snow and right to four
decimals on alpine snow — and nothing but the `truncated` column distinguishes
the two. That column is the whole reason it is reported.

The 3-D section took 241.7 minutes of P100 time: 109 minutes each for the two
visible clean cases, 20 and 6 for the alpine ones.
