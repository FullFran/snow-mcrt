# From radiative transfer to remote sensing

A radiative transfer model gives albedo as a function of wavelength. A
satellite gives a handful of numbers. This document is about the step between
them, why it is not a formality, and what the physics of snow looks like once
an instrument has integrated over it.

Every number below is produced by
[`tests/test_sensor.py`](../tests/test_sensor.py) from the committed reference
curves in `data/reference/`. Nothing here is quoted from a paper without the
model reproducing it.

---

## 1. What a band actually reports

An instrument does not sample the albedo at a wavelength. It integrates it
against the spectral response of a detector, weighted by how much light there
was at each wavelength to begin with:

$$
\alpha_b = \frac{\int \alpha(\lambda)\, R_b(\lambda)\, E(\lambda)\, d\lambda}
                {\int R_b(\lambda)\, E(\lambda)\, d\lambda}
$$

- $R_b$ — the band's **spectral response**: what the detector can see.
- $E$ — the incident **spectral irradiance**: how much light there was to see with.

**Both weights matter and they are not the same thing.** Dropping $E$ is a fair
approximation for a narrow visible band, where the solar spectrum barely moves
across the band, and a poor one for a wide near-infrared band, where it moves a
lot. `band_albedo()` therefore takes the irradiance as an explicit optional
argument instead of quietly assuming it flat — an assumption that is fine until
it is not, and that should be visible in the call.

### On top-hat responses

A real response function is a measured curve with sloped shoulders and
out-of-band leakage, published by the agency that flies the instrument. What
ships here is the rectangular idealisation implied by any table that quotes a
centre and a width.

**A repository that ships an approximation owes the reader its size**, so
`response_shape_sensitivity()` measures it: it swaps each top-hat for a
Gaussian of the same full width at half maximum and reports the difference.

| Sentinel-2 band | Width (nm) | \|Gaussian − top-hat\| |
|---|---|---|
| B2 (490 nm) | 65 | 0.00038 |
| B3 (560 nm) | 35 | 0.00031 |
| B4 (665 nm) | 30 | 0.00040 |
| B8 (842 nm) | 115 | 0.00096 |
| B8A (865 nm) | 20 | 0.00087 |
| B11 (1610 nm) | 90 | 0.00002 |
| **B12 (2190 nm)** | **180** | **0.00413** |

The pattern is the expected one. The shape of the response barely matters where
the albedo curve is flat across the band, and matters most in B12 — the widest
band, sitting where ice absorption turns sharply within the band's own width.
Even there the error is 0.4%, an order of magnitude below the spread between
radiative transfer models in the literature, so the idealisation is fit to
ship. `Tabulated` exists to replace it with a measured curve when it is not.

---

## 2. The spectral signature of snow

Clean snow, grain radius 100 µm, integrated over Sentinel-2 MSI:

| Band | Centre (nm) | Albedo |
|---|---|---|
| B2 | 490 | 0.992 |
| B3 | 560 | 0.980 |
| B4 | 665 | 0.956 |
| B8 | 842 | 0.872 |
| B8A | 865 | 0.862 |
| B11 | 1610 | **0.055** |
| B12 | 2190 | 0.065 |

**A factor of eighteen between the green band and the shortwave infrared.**
That is the whole signature, and it comes from one fact: the imaginary part of
the refractive index of ice is about seven orders of magnitude larger at
1600 nm than at 500 nm (Warren & Brandt 2008). In the visible a photon can
scatter thousands of times before being absorbed, so almost everything comes
back out. In the shortwave infrared it is absorbed after a handful of
scatterings.

---

## 3. Why NDSI works

$$
\mathrm{NDSI} = \frac{\alpha_\text{vis} - \alpha_\text{swir}}
                     {\alpha_\text{vis} + \alpha_\text{swir}}
$$

| Instrument | Bands | NDSI, clean snow |
|---|---|---|
| Sentinel-2 MSI | B3 / B11 | **0.894** |
| MODIS | b4 / b6 | **0.881** |

Operational snow mapping thresholds at NDSI > 0.4 (Hall et al. 1995). Clean
snow clears it by a wide margin.

**Why this pair and not brightness.** Nothing else common in a scene is bright
in one band and dark in the other. Cloud is bright in both; rock and soil are
middling in both. The *contrast* between the two is therefore a snow detector
rather than a brightness detector, which is why the index survived from
Landsat TM (Dozier 1989) into the operational MODIS product.

The two instruments differ by 0.013. Same physics, different bands: exactly the
kind of small, non-zero disagreement that tells you the integration step is
doing something real.

---

## 4. The two parameters are almost orthogonal in band space

This is the result worth stopping on.

### Grain size lives in the shortwave infrared

| Grain radius | B3 (560 nm) | B11 (1610 nm) |
|---|---|---|
| 50 µm | 0.986 | 0.110 |
| 100 µm | 0.980 | 0.055 |
| 250 µm | 0.969 | 0.020 |
| 500 µm | 0.956 | 0.009 |
| 1000 µm | 0.938 | 0.006 |

B11 falls by a factor of **eighteen**. B3 moves by 5%.

The mechanism: in a weakly absorbing medium the albedo depends on grain size
only through the number of scatterings a photon survives, which barely changes
when almost none are absorbed. Where absorption is strong, the photon's path
inside a single grain matters, and that path scales with grain radius. Larger
grains absorb more per encounter. Hence the SWIR reads size and the visible
does not (Wiscombe & Warren 1980; Nolin & Dozier 2000).

### Black carbon lives in the visible

| Black carbon | B3 (560 nm) | B11 (1610 nm) |
|---|---|---|
| 0 ng/g | 0.980 | 0.0550 |
| 1 ng/g | 0.979 | 0.0550 |
| 10 ng/g | 0.974 | 0.0550 |
| 100 ng/g | 0.946 | 0.0550 |
| 1000 ng/g | **0.848** | **0.0550** |

**B11 does not move at all.** Not approximately — to four decimal places,
across three orders of magnitude of soot loading.

The reason is a competition the numbers make obvious. Black carbon absorbs
broadly, but in the shortwave infrared *the ice itself already absorbs
overwhelmingly more*, so adding soot changes nothing measurable. In the
visible, ice is nearly transparent and even a trace of a strong absorber
dominates the budget (Warren & Wiscombe 1980).

### The mechanism, from Mie to the band

The tables above are a correlation until the chain behind them is written down.
It is short, and it is the canonical one — grain radius enters through Mie
theory, which returns the three numbers transport actually consumes:

```
radius r  ──Mie──▶  ω (single-scattering albedo)
                    g (asymmetry)                ──▶  transport  ──▶  albedo
                    σ_ext (extinction)
```

Computed from `data/ice/warren_brandt_2008.dat` at snow density 300 kg/m³:

**560 nm — ice is nearly transparent, `k = 2.8 × 10⁻⁹`**

| Radius | 1 − ω | σ_ext (m⁻¹) | Absorption length 1/(σ·(1−ω)) |
|---|---|---|---|
| 50 µm | 4.2 × 10⁻⁶ | 6681 | **35.95 m** |
| 100 µm | 8.0 × 10⁻⁶ | 3326 | **37.55 m** |
| 250 µm | 2.0 × 10⁻⁵ | 1328 | **37.91 m** |
| 500 µm | 4.0 × 10⁻⁵ | 662 | **38.01 m** |
| 1000 µm | 7.9 × 10⁻⁵ | 331 | **38.07 m** |

**Two effects cancel.** The co-albedo grows roughly in proportion to the
radius, because absorption within a single grain scales with the path through
it. The extinction coefficient falls roughly as its inverse, because at fixed
density larger grains means fewer of them. Their product — which is what sets
how far a photon travels before being absorbed — barely moves: 36 to 38 metres
across a factor of twenty in radius.

That cancellation *is* the reason the visible band cannot size a grain. It is
not that the band is insensitive by construction; it is that the physics
conspires to make the medium look the same.

**1610 nm — ice absorbs, `k = 2.7 × 10⁻⁴`**

| Radius | 1 − ω | Absorption length |
|---|---|---|
| 50 µm | 0.112 | **1.3 mm** |
| 100 µm | 0.192 | 1.6 mm |
| 250 µm | 0.331 | 2.3 mm |
| 500 µm | 0.419 | 3.6 mm |
| 1000 µm | 0.460 | **6.6 mm** |

Here the cancellation breaks. The co-albedo saturates towards a limit rather
than growing linearly, so it no longer offsets the falling extinction, and the
absorption length grows fivefold with grain size. Larger grains mean a longer
path inside absorbing ice, which means a darker snowpack.

**That is the whole factor of eighteen in B11**, derived rather than observed.

### Why this matters

The two parameters a snow retrieval wants act on **different bands**. That is
what makes a two-parameter inversion well-posed rather than degenerate: the
Jacobian is close to diagonal, so the retrieval is not trying to separate two
effects that look alike.

It is also why NDSI cannot do the job. The index does respond to grain size —
0.80, 0.89, 0.96, 0.98, 0.99 across the five radii — but it **saturates**: the
first step is worth 0.10 and the last 0.007. A quantity whose sensitivity
collapses by a factor of fifteen across the range of interest is a poor thing
to invert. Grain size is retrieved from the shortwave band directly, not from
an index built on top of it.

---

### What is standard here and what is not

The chain above — Mie for single scattering, then transport — is the canonical
approach and has been since Wiscombe & Warren (1980). SNICAR, TARTES and
BioSNICAR all do exactly this. **That is a feature.** A novel physics chain in
a snow albedo model would more likely be a wrong one.

What this repository does differently is not the physics:

- **Full three-dimensional Monte Carlo transport** rather than two-stream or
  delta-Eddington. The operational models approximate the transport; here it is
  the reference, and the approximations are what get checked against it.
- **A Fresnel surface and a buried object in three dimensions**, which is
  outside what plane-parallel models can express at all.
- **Cross-validation in two directions**: against van de Hulst in closed form
  and against TARTES as an independent implementation, with the provenance of
  every published curve committed under `data/reference/`.
- **A grain size distribution by default** (`grain_sigma_g = 1.5`, not 1).
  Monodisperse spheres support morphology-dependent resonances that spike
  absorption more than tenfold at isolated wavelengths, producing spectral
  features no real snowpack has.

## 5. Why no surrogate model, no emulator, no neural network

A neural surrogate of this forward model was considered and **rejected on
evidence**. The reasoning is worth recording, because "add ML" is the reflex.

1. **There is already a closed form.** `domain/analytic.py` implements
   van de Hulst's semi-infinite albedo, accurate to about 1% across the whole
   single-scattering-albedo range. Training a network to approximate it would
   produce a worse approximation of something already solved, and slower to
   build.
2. **The niche is occupied.** A published emulator of BioSNICAR already
   predicts spectral snow albedo from grain size, liquid water content and
   light-absorbing particle loading, presented as a lightweight alternative to
   the full radiative transfer model inside an inversion scheme. In the
   microwave domain, DMRT-Bic-NN reduced runtime from ~20.1 hours to ~0.54
   seconds. This is a mature line of work, not an opening.
3. **The premise does not hold here.** A surrogate earns its place when the
   forward model is the bottleneck. It is not: the analytic path sweeps a few
   hundred wavelengths in a second. Every table in this document was produced
   in less time than it takes to load a training set.

**The inverse direction is tractable for the same reason the surrogate is
pointless.** A retrieval needs many forward evaluations; the forward model is
already cheap; therefore the retrieval can be done directly, without an
emulator, a GPU or a training run.

---

## References

Ordered by what they support above.

**Optical constants and the signature**

- Warren, S. G., & Brandt, R. E. (2008). Optical constants of ice from the
  ultraviolet to the microwave: A revised compilation. *Journal of Geophysical
  Research: Atmospheres*, 113, D14220.
  DOI: [10.1029/2007JD009744](https://doi.org/10.1029/2007JD009744)
  — the ice refractive index this repository uses, in `data/ice/`.
- Warren, S. G. (1982). Optical properties of snow. *Reviews of Geophysics*,
  20(1), 67–89. DOI: [10.1029/RG020i001p00067](https://doi.org/10.1029/RG020i001p00067)

**Grain size and impurities**

- Wiscombe, W. J., & Warren, S. G. (1980). A model for the spectral albedo of
  snow. I: Pure snow. *Journal of the Atmospheric Sciences*, 37(12), 2712–2733.
  — why the SWIR reads grain size.
- Warren, S. G., & Wiscombe, W. J. (1980). A model for the spectral albedo of
  snow. II: Snow containing atmospheric aerosols. *Journal of the Atmospheric
  Sciences*, 37(12), 2734–2745.
  — why impurities act in the visible and not in the SWIR.
- Nolin, A. W., & Dozier, J. (2000). A hyperspectral method for remotely
  sensing the grain size of snow. *Remote Sensing of Environment*, 74(2),
  207–216. DOI: [10.1016/S0034-4257(00)00111-5](https://doi.org/10.1016/S0034-4257\(00\)00111-5)
  — the retrieval this document's band tables set up.

**The index and the instruments**

- Dozier, J. (1989). Spectral signature of alpine snow cover from the Landsat
  Thematic Mapper. *Remote Sensing of Environment*, 28, 9–22.
  DOI: [10.1016/0034-4257(89)90101-6](https://doi.org/10.1016/0034-4257\(89\)90101-6)
  — where the visible/SWIR contrast becomes an index.
- Hall, D. K., Riggs, G. A., & Salomonson, V. V. (1995). Development of methods
  for mapping global snow cover using Moderate Resolution Imaging
  Spectroradiometer data. *Remote Sensing of Environment*, 54(2), 127–140.
  DOI: [10.1016/0034-4257(95)00137-P](https://doi.org/10.1016/0034-4257\(95\)00137-P)
  — the operational NDSI > 0.4 threshold.
- Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission
  for GMES operational services. *Remote Sensing of Environment*, 120, 25–36.
  DOI: [10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026)
  — the band definitions.
- Painter, T. H., et al. (2009). Retrieval of subpixel snow-covered area, grain
  size, and albedo from MODIS. *Remote Sensing of Environment*, 113(4),
  868–879. DOI: [10.1016/j.rse.2009.01.001](https://doi.org/10.1016/j.rse.2009.01.001)
  — an operational multi-parameter retrieval built on this separability.

**On the surrogate question**

- Flanner, M. G., et al. (2007). Present-day climate forcing and response from
  black carbon in snow. *Journal of Geophysical Research*, 112, D11202.
  DOI: [10.1029/2006JD008003](https://doi.org/10.1029/2006JD008003)
  — SNICAR, the lineage the published emulator sits on.

---

## What is not here yet

Stated plainly, because a document that only lists what works is advertising.

- **No solar zenith angle.** The engine handles collimated and diffuse
  incidence; a real retrieval needs the dependence on illumination geometry.
- **No BRDF.** Snow is not Lambertian, and at the grazing angles common in
  polar and alpine scenes the anisotropy is large.
- **No atmosphere.** These are surface albedos. A satellite sees
  top-of-atmosphere reflectance, and getting from one to the other is its own
  correction.
- **No measured response functions.** Top-hat idealisations, with the error
  quantified in section 1.
- **No broadband albedo.** Solar-spectrum-weighted integration is the quantity
  energy balance models want, and it needs an irradiance spectrum this
  repository does not yet commit.
