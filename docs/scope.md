# Project scope

## Research questions for v1

1. Reproduce Wiscombe & Warren (1980) spectral albedo curves for pure snow across grain sizes (≈50 μm to 1 mm).
2. Reproduce Warren & Wiscombe (1980) effect of black carbon (1–1000 ng/g) on visible albedo.
3. Validate e-folding depth against Libois et al. (2013) measurements and TARTES outputs.
4. Cross-check against SNICAR for matching parameter sets.

## Deliverables

- **Engine** — vectorized Python MC core, deterministic seeded runs, full unit-test coverage of physics kernels.
- **Notebooks** — one per research question, reproducing figures from the source paper.
- **Data layer** — curated ice optical constants and impurity absorption spectra, each file sourced and dated.
- **Validation suite** — automated comparison against tabulated benchmarks, with tolerance thresholds documented.

## Non-goals (v1)

- Non-spherical grain morphology (geometric optics for large grains, IRR for columnar ice).
- Snow metamorphism / microstructure evolution.
- Coupled atmospheric radiative transfer.
- GPU acceleration — profile first, decide later.

## Success criteria

- All four benchmark reproductions within published error bars.
- CI green on every push.
- A reader can reproduce any figure from a fresh clone in under 10 minutes.
