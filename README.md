# snow-mcrt

Monte Carlo Ray Tracing for light propagation in snow.

**Status:** early stage — scope and bibliography only. Engine not yet started.

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

## Planned layout

```
snow-mcrt/
├── docs/
│   ├── bib/
│   │   └── references.md
│   └── scope.md
├── python/           # MC engine (core + FastAPI service)
├── notebooks/        # One notebook per benchmark reproduction
├── data/             # Ice optical constants, published benchmarks
└── tests/
```

## References

See [`docs/bib/references.md`](docs/bib/references.md). Scope detail in [`docs/scope.md`](docs/scope.md).

## License

TBD. Likely MIT or Apache-2.0 once the engine is public.
