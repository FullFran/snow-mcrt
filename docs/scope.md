# Project scope

## Research questions for v1

| # | question | status |
| - | -------- | ------ |
| 1 | Reproduce Wiscombe & Warren (1980) spectral albedo curves for pure snow across grain sizes (≈50 μm to 1 mm). | curves computed from Warren & Brandt (2008) constants and committed under `data/reference/`; **not yet compared point by point against the published tables** |
| 2 | Reproduce Warren & Wiscombe (1980) effect of black carbon (1–1000 ng/g) on visible albedo. | curves computed and committed; magnitude matches, digitised comparison pending |
| 3 | Validate e-folding depth against Libois et al. (2013) measurements and TARTES outputs. | computed and plotted; no comparison against measurement yet |
| 5 | Establish where the diffusion approximation the detectability work rests on is valid, using 3-D transport as the reference. | **done.** Four snowpacks over three decades of co-albedo, 2e6 photons each on a P100: diffusion holds to 9-11% over 3-12 transport mean free paths and 5-9% over 12-50. Inside 3 it is inapplicable, not inaccurate. See `docs/detectability.md` and `results/kaggle/` |
| 4 | Cross-check against SNICAR and TARTES for matching parameter sets. | **TARTES done and decomposed**: our radiative transfer reproduces it to 0.0023 worst case across five grain sizes once the grain model is held fixed. The residual 0.11 is the sphere assumption. SNICAR not started |

"Computed" is not "validated". Until each curve is compared against the
published numbers with a stated tolerance, these are self-consistent results,
not reproductions.

Which solver produced which result is part of that claim and is easy to lose:
questions 1-3 above are **analytic** curves (van de Hulst with similarity
scaling), question 4 compares the **analytic** solver against TARTES, and
question 5 is the only one whose numbers come from the **Monte Carlo** engine
this project is named after. The engine is validated -- against van de Hulst
in one dimension, and against the plane-parallel engine in three -- but it has
so far produced far less of the published output than the name suggests.

## Deliverables

- **Engine** — vectorized Python MC core, deterministic seeded runs, full unit-test coverage of physics kernels.
- **Notebooks** — one per research question, reproducing figures from the source paper.
- **Data layer** — curated ice optical constants and impurity absorption spectra, each file sourced and dated.
- **Validation suite** — automated comparison against tabulated benchmarks, with tolerance thresholds documented.

## Non-goals (v1)

- Non-spherical grain morphology (geometric optics for large grains, IRR for columnar ice).
- Snow metamorphism / microstructure evolution.
- Coupled atmospheric radiative transfer.

GPU acceleration has moved *into* scope for v1. The array-backend port is in
place from the start with both NumPy and CuPy adapters, because the port is
what keeps `domain/` free of any array-library import and what makes the NumPy
oracle possible — it is a correctness device before it is a performance one.
The GPU is expected to pay only in the photon-transport loop; Mie evaluation
over a wavelength grid stays on the host. See `docs/architecture.md`.

## Success criteria

- All four benchmark reproductions within published error bars.
- CI green on every push.
- A reader can reproduce any figure from a fresh clone in under 10 minutes.
