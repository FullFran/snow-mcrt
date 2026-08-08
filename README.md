# snow-mcrt

[![CI](https://github.com/FullFran/snow-mcrt/actions/workflows/ci.yml/badge.svg)](https://github.com/FullFran/snow-mcrt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Monte Carlo photon transport for light propagation in snow, validated against
closed-form radiative transfer and the canonical snow-optics literature.

## How this started

I spent a while doing optical materials design — ray tracing through disordered
media, where light does not travel in straight lines and every useful answer
comes out of a scattering calculation rather than a geometric one.

I missed it. And a question kept nagging: **could the same machinery find
things buried inside a material?** If light diffuses through a disordered
medium and comes back out, it has been somewhere, and what it met on the way is
encoded in how it returns.

Then a second thought, which is what actually made this a project. Snow is a
disordered random medium too — ice grains packed at random, light scattering
thousands of times before it escapes or is absorbed. Same physics, a medium
anyone can walk on, and a literature going back forty years with hard numbers
to check against.

So: **can I simulate light in snow, correctly enough that the answer would be
believed?** That question comes first, because the buried-object idea is
worthless without it. An engine that cannot reproduce a published albedo curve
has no business predicting what is under the surface.

This repository is the answer to the first question. And the second one is no
longer only scoped: there is now a 3-D engine with a real Fresnel surface and
objects buried in the snow, so it has a measured answer.

![Detection contrast against burial depth](docs/figures/detect-transport-depth.png)

A 20 cm slab removes **53%** of the returned light when its top sits a tenth
of a penetration depth down, 8.5% at six tenths, and 2.4% at one — a fall of
roughly a factor of six per penetration depth, which is the two-way
attenuation the design note argued for, now measured instead of assumed.

The surprise is the lower curve. **A cavity absorbs nothing at all and still
removes a third as much light as a black slab.** Inside a void there is nothing
to scatter from, so a photon that enters runs straight to the far wall well
below the depth anything returns from: a light pipe pointing away from the
detector.

The shaded band is where the answer stops being physics and becomes photon
budget. It falls as `1/sqrt(N)`, which is why the deep cases run on a GPU.
Full reasoning in [`docs/detectability.md`](docs/detectability.md).

## Light in snow, from first principles

![Spectral albedo of pure snow](docs/figures/spectral-albedo-grain-size.png)

Snow is brilliant white to the eye and dark in the near infrared, and both
facts come out of the same calculation. Nothing above is fitted: Mie theory on
ice grains, measured optical constants, a radiative transfer solution. The
absorption bands at 1030, 1250, 1500 and 2000 nm are not put in by hand — they
fall out of the ice.

## What this is

A clean, testable engine for the physics of light in snow. It answers four
questions, each checked against something independent:

1. **Spectral albedo of pure snow** across grain sizes — Wiscombe & Warren (1980).
2. **The effect of absorbing impurities** — Warren & Wiscombe (1980).
3. **How deep light penetrates** — e-folding depth, against field measurement.
4. **Cross-checks** against SNICAR and TARTES for matching parameters.

Everything is validated. Where a closed-form solution exists, the engine must
reproduce it; where an exact invariant holds, it must hold to floating point.

## What it found

Building it turned up four things worth stating on their own.

**The two-stream albedo is not accurate enough to validate a Monte Carlo.**
It runs 18.7% high at `omega = 0.5` and 8.9% at 0.9, converging only in the
conservative limit. The transport matches van de Hulst's fit to the exact
solution to a fraction of a percent. Judging the engine against two-stream
would have meant chasing a bug that was the yardstick all along.

**A uniform `mu` grid silently destroys the Mie phase function.** At snow size
parameters the forward diffraction peak spans microradians. A uniform
20 001-point grid steps over it and produces a "phase function" that integrates
to 20.4 with a mean cosine above one. Nothing raises at the point of use.

**Monodisperse spheres are an actively misleading idealisation.** A perfect
sphere supports resonances that spike absorption 13-fold at isolated
wavelengths, putting spectral features in the albedo curve that no real
snowpack has. A 5% spread in grain size destroys them. The default here is a
log-normal distribution, not a single radius.

**Russian roulette conserves energy only in expectation.** Residuals are
sign-changing at the 1e-6 level; with roulette off the ledger closes to float64
epsilon. That is an unbiased estimator behaving correctly, not a leak, and the
tests assert the two cases separately.

## Results

**Which solver produced which curve**, because the name of the repository
makes one guess and it is the wrong one. The spectral albedo curves below are
**analytic** — van de Hulst with similarity scaling, accurate to about 1% and
fast enough to sweep hundreds of wavelengths in a second. The **Monte Carlo**
engine is the reference that validates them, not the thing that draws them: it
spot-checks the analytic curve at a handful of wavelengths, and it produced the
diffusion validity range further down. Sweeping a visible curve with it is not
a slow run but an infeasible one — clean snow at 450 nm needs of order three
million scattering orders per photon.

Both are validated: the analytic solver against TARTES to 0.0023, the Monte
Carlo against van de Hulst in one dimension and against the plane-parallel
engine in three. Which one drew a given figure is recorded in its manifest
under `data/reference/`.

### Impurities

![Black carbon in snow](docs/figures/black-carbon.png)

One part per billion of black carbon is already visible. A hundred takes
visible albedo from 0.99 to the mid 0.95s — reproducing Warren & Wiscombe
(1980) with no tuning. Mineral dust needs roughly a hundred times the mass for
comparable darkening, which is why the field quotes carbon in ppb and dust in
ppm.

### What a satellite would see

The model gives albedo against wavelength. An instrument gives a handful of
numbers, each an integral against a detector's spectral response. Integrating
over Sentinel-2 MSI and MODIS closes that gap, and the result is the spectral
signature of snow in two numbers:

| | B3 (560 nm) | B11 (1610 nm) | NDSI |
| --- | --- | --- | --- |
| Clean snow, r = 100 um | 0.980 | 0.055 | **0.894** |

A factor of eighteen between green and shortwave infrared. That contrast, and
not brightness, is what makes snow separable from cloud and rock, and it is why
the operational threshold of NDSI > 0.4 works.

The two parameters a retrieval wants turn out to act on **different bands**.
Grain size moves B11 by a factor of eighteen from 50 to 1000 um while moving
B3 by 5%. Black carbon takes B3 from 0.980 to 0.848 across three orders of
magnitude of loading while leaving B11 at 0.0550 — unchanged to four decimal
places. The Jacobian is close to diagonal, which is what makes a
two-parameter inversion well posed rather than degenerate.

The top-hat band approximation this ships with is measured rather than
asserted: swapping each band for a Gaussian of matched full width at half
maximum moves the answer by at most 0.004, in the widest band.

Bands, index, separability, the papers behind each claim, and an explicit list
of what is still missing: [`docs/remote-sensing.md`](docs/remote-sensing.md).

### What varies, and how deep light reaches

![Co-albedo and penetration depth](docs/figures/co-albedo-and-penetration.png)

The single-scattering albedo `omega` sits within a millionth of unity across
the visible, so `1 - omega` is the quantity that actually moves — and it spans
six orders of magnitude. That is why visible albedo is an impurity diagnostic
while the near infrared is a grain-size one.

### The ice underneath it all

![Absorption of pure ice](docs/figures/ice-absorption.png)

Everything above follows from this curve. Note the shaded band: below 390 nm
the tabulated `k` is a reported *upper limit*, not a measurement, so
penetration depths computed there are bounds rather than predictions.

## Validated against an independent implementation

![Cross-validation against TARTES](docs/figures/tartes-validation.png)

Two codes producing similar curves proves very little. The interesting question
is *where* they differ, so the comparison against TARTES (Libois, Picard et
al., 2013) is decomposed rather than reported as a single number.

Both codes read the same Warren & Brandt (2008) constants — verified, not
assumed. Then TARTES's own single-scattering parameters are fed into **our**
radiative transfer solver. If the two then agree, they solve the transfer
problem the same way and everything else between them is a modelling choice.

| grain radius | transfer residual | grain-model residual |
| ------------ | ----------------- | -------------------- |
| 50 μm        | 0.00020           | 0.10087              |
| 100 μm       | 0.00051           | 0.11388              |
| 250 μm       | 0.00128           | 0.11475              |
| 500 μm       | 0.00197           | 0.11454              |
| 1000 μm      | 0.00231           | 0.11373              |

**The transfer solution agrees to better than 0.003. The remaining 0.11 is
entirely the grain model** — a factor of fifty between them.

And that residual has a name. Full Mie on spheres gives `g ≈ 0.89`; TARTES uses
`0.82`, calibrated for the non-spherical grains real snow is actually made of.
Spheres over-predict forward scattering, photons therefore travel deeper per
collision and accumulate more path in ice, and our albedo sits below TARTES
everywhere the ice absorbs at all. In the blue, where absorption nearly
vanishes, the two converge regardless.

Non-spherical morphology is explicitly out of scope for v1. This measures that
limitation rather than hiding it.

## Architecture

Clean/hexagonal, and each port earns its place:

```
src/snow_mcrt/
  domain/        physics; imports no array library and no numerical package
    optics.py      complex refractive index, units and sign conventions
    mie.py         single-scattering properties, grain size distributions
    phase.py       Henyey-Greenstein and tabulated phase functions
    medium.py      snowpack layers, impurities, external mixing
    transport.py   vectorised Monte Carlo photon transport
    analytic.py    closed-form solutions -- the oracle
  ports/         backend (NumPy/CuPy), Mie solver, optical data
  adapters/      NumPy is the oracle, not a fallback; CuPy; miepython
                 plus a caching decorator over the Mie port
  application/   one use case per research question
  infra/         CSV and manifest writers
scripts/         run (headless, draws nothing) and plot (never simulates)
data/ice/        optical constants, with provenance
data/mie/        committed Mie table, stamped with what produced it
data/reference/  committed results, so figures reproduce without re-running physics
```

Photons are **an axis of the array**, not an iteration: the transport loop runs
once per scattering order and every live photon steps together. The backend
port keeps `domain/` free of any array-library import, which is what makes the
NumPy oracle possible — and it is why the same physics will run on a GPU.

Full rationale, the conventions that are silent when wrong, and the measured
case for the GPU: [`docs/architecture.md`](docs/architecture.md).

## Running on a GPU

The CuPy adapter is validated on real hardware — results committed under
[`results/kaggle/`](results/kaggle/), produced on a Tesla P100.

Against the NumPy oracle, agreement tightens as the medium gets brighter:
0.50% at `omega = 0.5`, 0.053% at 0.9, 0.016% at 0.95. More photons survive
longer, so the estimator has more to work with.

The timing is worth stating honestly, because it is not a straight win:

| `omega` | NumPy | CuPy | speedup |
| ------- | ----- | ---- | ------- |
| 0.50 | 0.31 s | 3.68 s | **0.08x** |
| 0.90 | 1.26 s | 0.21 s | 6.0x |
| 0.95 | 2.94 s | 0.36 s | 8.2x |

At `omega = 0.5` the transport finishes in two dozen scattering orders and
kernel launch overhead dominates completely — the GPU is twelve times
*slower*. It pays in deep transport and nowhere else, which is the same
conclusion the architecture reached from the other direction.

Deep transport at 500 000 photons reproduces the analytic solution to 0.02%,
0.08% and 0.14% at 700, 900 and 1100 nm, with `truncated = 0` throughout. The
`1/(1 - omega)` cost law comes out measured rather than asserted: a factor of
37 in co-albedo buys 33 times the scattering orders and 32 times the wall
clock.

Those Monte Carlo values are bit-identical across two separate GPU runs, since
the seed is part of the configuration. The *timings* moved by up to 20% — a
shared GPU is the one thing here that does not reproduce, which is why the two
are reported separately.

### Where the diffusion approximation holds

![Where diffusion holds](docs/figures/detect-diffusion-validity.png)

The buried-object work rests on diffusion theory, and until the 3-D engine
existed there was nothing to check it against. Measured at two million photons
per case over four snowpacks spanning three decades of co-albedo, **diffusion
is good to 9–11% over 3–12 transport mean free paths** and 5–9% out to 50.
Inside about three it is not inaccurate but inapplicable — the photon has not
yet forgotten which way it was going.

The shape is not the obvious one. The ratio dips below one at the source,
crosses it around 3–7 `mfp'`, peaks near 1.03, and settles about 7% low
further out: diffusion errs in one direction near the source and the other
beyond. Beyond 50 `mfp'` the curves are drawn dotted because there the
comparison measures photon budget rather than diffusion.

```bash
kaggle kernels push -p kaggle
kaggle kernels output fran17/snow-mcrt-gpu-validation -p results/kaggle
```

## Getting started

```bash
uv venv && uv pip install -e '.[dev]'
pytest                                    # 371 passed, 4 skipped without CUDA

python scripts/run_albedo.py --output data/reference     # compute, draw nothing
python scripts/plot_albedo.py --output docs/figures      # draw, compute nothing
```

That split is deliberate. `run_albedo.py` opens no windows, so it runs
unchanged in a batch job; `plot_albedo.py` reads committed CSVs, so a figure
can be restyled without waiting on any physics. Every run writes a manifest
recording every parameter that produced it, seed included.

Mie evaluation at snow size parameters is the expensive part, and it repeats:
of the 19 578 `(m, x)` points the detectability figures request, only 1 258 are
distinct. Every run script memoises them, keyed on the exact bits of the input
so a hit can only ever return what the solver itself would have. `run_albedo.py`
goes from 237 s to 0.6 s on a rerun, with the output CSVs byte-for-byte
identical. Pass `--no-cache` to evaluate everything from scratch.

Two tables, because there are two jobs. `data/mie/` is **committed**: it covers
the configurations behind the published results, so cloning is enough to vary
one of those runs without first spending four minutes of CPU re-deriving
settled numbers. It is regenerated deliberately, never as a side effect:

```bash
python scripts/build_mie_cache.py            # rebuild it
python scripts/build_mie_cache.py --check    # does it still cover every published run?
```

The working cache lives in `$SNOW_MCRT_CACHE_DIR` (default
`~/.cache/snow-mcrt`) and stays out of the tree — `npz` is compressed binary
that git cannot delta, so a mutable table under version control would rewrite
itself on every run.

Every table records the solver, its version, NumPy, and the platform, and a
stamp that does not match is **refused and recomputed**. Without that, a reader
on a newer `miepython` would get a bit-exact *hit* and walk away with someone
else's numbers believing they were their own — a cache that can silently mask a
change of solver is a way to invalidate a validation. That is also why
`miepython` is pinned exactly rather than floated.

CuPy is optional (`pip install -e '.[gpu]'`). Without it the GPU tests skip and
everything else runs.

## Physics scope — v1

- Multiple scattering by ice grains via Mie theory, over a log-normal size
  distribution.
- Ice optical constants from Warren & Brandt (2008), 44 nm to 2 m.
- Henyey–Greenstein and full-Mie phase functions.
- Absorbing impurities (black carbon, mineral dust) as mixing ratios, externally
  mixed.
- Semi-infinite and finite plane-parallel slabs.
- GPU acceleration through the backend port.

**Out of scope in v1:** non-spherical grain morphology, internal mixing of
impurities, coherent effects, polarization, 3-D geometry, snow metamorphism,
coupled atmospheric radiative transfer.

## Back to the original question

Can this find objects buried in snow? [`docs/detectability.md`](docs/detectability.md)
scopes it properly. The short version, computed with the engine above:

**The depth limit is set by how clean the snow is, not by the optical
properties of ice.** In the blue, where ice is at its most transparent, pure
ice alone would let light reach nearly half a metre. One nanogram per gram of
black carbon — one part per billion, cleaner than almost anywhere on Earth —
cuts that to 10 cm.

Two more results that changed the shape of the problem:

- The diffuse sensitivity kernel peaks at **`rho/3`**, not the `rho/2` of the
  tissue-optics rule of thumb. Probing 30 cm needs a metre-scale baseline.
- **Photon budget is not the constraint.** Even at a 150 cm separation a
  half-millijoule pulse delivers eighteen times the photons a 1% contrast
  measurement needs. The real limit is clutter — natural snowpack layering
  perturbs the diffuse field just as a buried object does — which turns the
  question from amplitude into separability.

It is the same lineage as diffuse optical tomography in tissue, and MCML, the
canonical Monte Carlo for photon transport in tissue, was already in this
project's bibliography before the connection was obvious.

Honest positioning: this does not compete with avalanche beacons. RECCO and
transceivers win at depth because dry snow is nearly transparent at radio
frequencies. Optical is capped at tens of centimetres and no engineering moves
that. Where it does stand up — detectability-limit theory, very clean polar
snow, and **voids rather than absorbers**, since a cavity perturbs diffusion far
more strongly than an absorber of the same size — is set out in the note.

## Known gaps

- The four benchmark notebook reproductions are not written yet.
- SNICAR cross-check not started. TARTES is done — see above.
- Clean snow at 500 nm (`1 - omega ~ 5e-6`) is still out of reach: it needs of
  order a million scattering orders, over an hour even on a P100.
- Grain morphology is spheres only, and the TARTES comparison measures exactly
  what that costs.

## References

[`docs/bib/references.md`](docs/bib/references.md). Scope detail in
[`docs/scope.md`](docs/scope.md). Data provenance in
[`data/ice/SOURCE.md`](data/ice/SOURCE.md).

## License

MIT — see [`LICENSE`](LICENSE).

The ice optical constants under `data/ice/` are the published Warren & Brandt
(2008) compilation and carry their own terms. See
[`data/ice/SOURCE.md`](data/ice/SOURCE.md) for citation and provenance.
