# snow-mcrt

Monte Carlo Ray Tracing for light propagation in snow.

**Status:** early stage. Single-scattering layer implemented and validated
against the Rayleigh and geometric limits. Photon transport not yet started.

## Goal

Reproduce canonical results from the snow-optics literature (spectral albedo curves, e-folding depth, grain-size and impurity effects) with a clean, testable Monte Carlo engine, and validate against published benchmarks (Warren & Wiscombe 1980, SNICAR, TARTES).

## Physics scope — v1

- Multiple scattering by spherical ice grains via Mie theory.
- Spectral absorption of ice from Warren & Brandt (2008) dataset.
- Henyey–Greenstein and full-Mie phase functions.
- Absorbing impurities (black carbon, mineral dust) as mixing ratios.
- 1-D slab geometry; layered snowpack as extension.

## Out of scope — v1

Non-spherical grain morphology, coherent effects, polarization, 3-D topography, snow metamorphism, coupled atmospheric radiative transfer.

## Layout

```
snow-mcrt/
├── docs/
│   ├── architecture.md   # why the layers are separated
│   ├── scope.md
│   └── bib/references.md
├── src/snow_mcrt/
│   ├── domain/           # physics; imports no array library
│   ├── ports/            # backend, Mie solver, optical data protocols
│   ├── adapters/         # NumPy (oracle) + CuPy; miepython; tabulated data
│   ├── application/      # one use case per research question
│   └── infra/            # CSV + manifest writers
├── notebooks/            # one per benchmark reproduction
├── data/reference/       # committed results, so figures reproduce without a GPU
└── tests/
```

Rationale for the layering, the two conventions that are silent when wrong,
and where the GPU actually pays: [`docs/architecture.md`](docs/architecture.md).

## Getting started

```bash
uv venv && uv pip install -e '.[dev]'
pytest
```

CuPy is optional (`pip install -e '.[gpu]'`). Without it, the GPU tests skip
and everything else runs.

## References

See [`docs/bib/references.md`](docs/bib/references.md). Scope detail in [`docs/scope.md`](docs/scope.md).

## License

TBD. Likely MIT or Apache-2.0 once the engine is public.
