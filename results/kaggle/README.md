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
| 0.50 | 0.33 s | 4.43 s | **0.07x** |
| 0.90 | 1.24 s | 0.19 s | 6.5x |
| 0.95 | 2.74 s | 0.37 s | 7.4x |

At `omega = 0.5` the transport finishes in about two dozen scattering orders
and kernel launch overhead dominates completely — the GPU is thirteen times
*slower*. This is worth stating plainly because it is the same conclusion the
architecture notes reached from the other direction: the GPU pays in deep
transport, not in anything that merely fits in an array.

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
