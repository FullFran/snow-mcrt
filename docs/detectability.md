# Detecting buried objects in snow — a design note

Where this project could go after v1, what the physics allows, and what the
engine would have to become. Written before any of it is built, so the
constraints are on the record rather than discovered halfway through.

**Status:** exploratory. No code exists for any of this. Every number below is
produced by the v1 engine or by diffusion theory.

**Updated with the real dataset.** An earlier version of this note used
order-of-magnitude placeholders for the ice absorption. Warren & Brandt (2008)
is now in `data/ice/`, and every depth has been recomputed against it. The
placeholders were wrong in both directions — 9x too optimistic at 450 nm, 3x
too pessimistic at 500 nm — so the numbers below supersede them. The
conclusions did not change; the magnitudes did.

## The question

> To what depth can an object buried in snow be detected optically, as a
> function of burial depth, object size and contrast, snow grain size,
> impurity loading, and wavelength — and what measurement geometry does it
> take?

Well-posed, falsifiable, and squarely answerable with a validated Monte Carlo
transport code. Note what it is not: it is not "build an avalanche beacon".
The answer may well be "23 cm", and that is a result rather than a failure.

## This problem already has a mature sibling

What is being described is **diffuse optical tomography**: reconstructing
inclusions inside a strongly scattering medium from light that has diffused
through it. The medical version — imaging haematomas, or cerebral blood
oxygenation through the skull — has three decades of theory behind it.

The bibliography already points there. Wang, Jacques & Zheng (1995), MCML, is
the canonical Monte Carlo for photon transport in multi-layered *tissue*. The
methodological reference chosen for the snow work is the one the neighbouring
field built its instruments on.

Snow is a *better* diffuser than tissue: `omega` closer to 1, comparable `g`.
Diffusion theory is more reliable here, not less, which means the analytic
machinery transfers with the approximations in better shape than the people
who developed them enjoyed.

## The regime

From the v1 engine, 100 um grains, 300 kg/m³:

| quantity                          | value    |
| --------------------------------- | -------- |
| scattering mean free path `l`     | 0.20 mm  |
| transport mean free path `l*`     | 1.83 mm  |
| reduced scattering `mu_s'`        | 548 m⁻¹  |
| diffusion coefficient `D`         | 0.61 mm  |

`l* / l = 1/(1-g) = 9`, so a photon needs about nine collisions to forget
where it was going. Any object deeper than a centimetre or so is being viewed
through a fully diffusive medium — there is no ballistic component to work
with, and every useful observable is a diffusion observable.

## What sets the depth limit — and it is not what you would guess

Ice is extraordinarily transparent in the blue. Its absorption minimum sits
near 390–400 nm, which is why deep glacial ice looks blue. Against pure ice
absorption alone, penetration is generous:

At 400 nm, where Warren & Brandt report the deepest genuinely *measured*
minimum, `k = 2.365e-11`:

| grain  | e-folding depth | depth at 1% two-way signal |
| ------ | --------------- | -------------------------- |
| 100 um | 36.0 cm         | 82.8 cm                    |
| 1 mm   | 114.0 cm        | 262.6 cm                   |

Better than a metre in coarse snow. Encouraging — and misleading.

(Below 390 nm the tabulated `k` flattens onto a reported *upper limit* of
2.0e-11 rather than a measurement, so anything computed there is a bound. 400 nm
is the shortest wavelength with a real number behind it.)

Because ice is *so* weakly absorbing there, trace impurities take over
completely:

100 um grains at 400 nm, 300 kg/m³:

| black carbon | `1 - omega` | e-folding depth | depth at 1% two-way signal |
| ------------ | ----------- | --------------- | -------------------------- |
| 0 ng/g       | 9.3e-08     | 36.0 cm         | 82.8 cm                    |
| 0.1 ng/g     | 1.8e-07     | 25.8 cm         | 59.4 cm                    |
| **1 ng/g**   | 9.7e-07     | **11.1 cm**     | **25.6 cm**                |
| 5 ng/g       | 4.5e-06     | 5.2 cm          | 11.9 cm                    |
| 20 ng/g      | 1.8e-05     | 2.6 cm          | 6.0 cm                     |
| 100 ng/g     | 8.8e-05     | 1.2 cm          | 2.7 cm                     |

**One nanogram per gram of black carbon cuts the useful depth from 36 cm to
11 cm.** One part per billion.

Coarse grains help but do not rescue it: 1 mm grains go from 114 cm clean to
35 cm at 1 ng/g.

And 1 ng/g is cleaner than almost anywhere on Earth. Remote Antarctic snow is
of order 0.1–0.3 ng/g; Arctic snow runs 5–50; mid-latitude mountain snow 10–100
(figures approximate, pending proper citation).

So the governing conclusion, and it is not obvious in advance:

> The optical depth limit in snow is set by how clean the snow is, not by the
> optical properties of ice.

This single fact should drive the whole research programme. It also means the
impurity machinery already built in v1 is not a side feature here — it is the
central variable.

## The sensitivity kernel

In a diffusive medium, a source and a detector separated by `rho` sample a
banana-shaped volume between them. The tissue-optics rule of thumb puts its
peak at `rho/2`. With snow's actual parameters, computed from the diffusion
Green's function with an extrapolated boundary:

| separation `rho` | peak sampling depth | ratio |
| ---------------- | ------------------- | ----- |
| 10 cm            | 3.2 cm              | 0.32  |
| 20 cm            | 6.6 cm              | 0.33  |
| 40 cm            | 13.2 cm             | 0.33  |
| 60 cm            | 19.3 cm             | 0.32  |
| 100 cm           | 30.2 cm             | 0.30  |
| 150 cm           | 41.9 cm             | 0.28  |

**`rho/3`, not `rho/2`**, and the ratio drifts downward once absorption starts
to bite past a metre. To interrogate 30 cm of snow you need source and detector
about a metre apart. That is a hard geometric constraint on any instrument, and
it is worth knowing before anyone designs a handheld device.

## Photon budget is not the limiting factor

Diffuse reflectance falls steeply with separation — five orders of magnitude
from 10 cm to 150 cm — but it falls from a very large number:

| `rho`  | probes | relative `R` | detected photons in 1 cm² from a 1e15-photon pulse |
| ------ | ------ | ------------ | -------------------------------------------------- |
| 10 cm  | 3.3 cm | 1.00         | 8.4e10                                             |
| 40 cm  | 13 cm  | 1.3e-02      | 1.1e09                                             |
| 100 cm | 33 cm  | 4.0e-04      | 3.3e07                                             |
| 150 cm | 50 cm  | 5.6e-05      | 4.6e06                                             |

A 1e15-photon pulse at 450 nm is about half a millijoule. Trivial.

Against shot noise, detecting a fractional contrast `C` at signal-to-noise `s`
needs `N >= (s/C)²` detected photons:

| contrast | SNR 5 requires    |
| -------- | ----------------- |
| 10%      | 2 500 photons     |
| 3%       | 27 778 photons    |
| 1%       | 250 000 photons   |

Even at `rho = 150 cm` there are 4.6e6 photons available — eighteen times what
a 1% contrast measurement needs.

**So the instrument is not photon-starved.** That kills the obvious engineering
objection and moves the real question elsewhere.

## The real limit is clutter, not noise

If shot noise is not the constraint, what is? Natural snowpack structure that
produces the same signal as the thing being looked for.

A real snowpack is layered. Density varies from 100 to 500 kg/m³ between
strata, grain radius from 50 um to over 1 mm across a melt-freeze crust or a
depth-hoar layer, and impurity loading varies with deposition history. Every
one of those perturbs the diffuse field. A buried object is a perturbation
competing against a background of perturbations.

This reframes the research question into its useful form:

> Not "can we detect a 1% contrast?" — we can. But: **is the object's signature
> separable from the snowpack's own?**

That is a question about *structure*, not amplitude, and it is what makes the
time-resolved measurement interesting rather than merely a refinement.

## Observables that carry depth information

**Spatially resolved reflectance `R(rho)`.** Vary the separation, sample
different depths. Simple; needs the metre-scale baselines computed above; badly
confounded by layering, since a horizontal layer and a buried object both change
`R(rho)` smoothly.

**Time-resolved `R(rho, t)`.** Pulse the source, histogram photon arrival
times. Late photons went deeper — the mapping from time to depth is far more
direct than the mapping from separation to depth, and the *shape* of the
temporal distribution discriminates a compact object from a horizontal layer in
a way that any steady-state measurement cannot. A layer shifts the whole curve;
a localised absorber removes a specific slice of path lengths.

In a Monte Carlo this costs **one extra accumulator**: total path length per
photon, already implicit in the free-path sampling. It is the cheapest large
capability available, and it should be added to the 1-D engine before the 3-D
work starts, where it is easy to validate against the analytic
time-domain diffusion solution.

**Frequency-domain.** Modulate the source, measure amplitude and phase shift.
Information-equivalent to time-resolved, easier instrumentation, worse depth
resolution. Worth noting; not worth building first.

## What the architecture would have to become

The v1 transport core is 1-D, plane-parallel and azimuthally symmetric. A
buried object breaks both symmetries. This is a genuine extension, not a
parameter change.

**Photon state grows.** From `(z, mu, weight)` to
`(x, y, z, ux, uy, uz, weight, pathlength)`. The azimuthal reduction that made
v1 cheap is gone — position and direction both become 3-vectors. Roughly triple
the array traffic per step, before any geometry.

**Ray-object intersection enters — the geometric kind.** Sphere, box, and
eventually a mesh. This is where ray tracing in the rendering sense genuinely
appears, having been absent from v1 despite the project's name.

**The air-snow interface stops being ignorable.** v1 lets photons enter freely,
which is fine when the observable is a hemispheric albedo. With a source and a
detector both sitting on the surface, Fresnel reflection and refraction at the
boundary shape the near-field response directly. `n = 1.31` gives an internal
reflection coefficient that materially changes the extrapolated boundary
condition — the `zb` already used in the Green's function above.

**Detectors become objects.** Spatial binning in `rho`, temporal binning in
path length, and an acceptance solid angle. v1 has no detector concept at all;
it counts everything that escapes.

**Cost lands squarely on the GPU.** v1 already measured 200 000 scattering
orders for clean visible snow on one CPU core. Three-dimensional geometry,
millions of photons, and a source-detector scan multiplies that. The backend
port exists precisely so this is a configuration change rather than a rewrite.

## How it stays honest

The discipline that made v1 trustworthy applies unchanged, and the extension
happens to come with unusually good oracles.

**The 1-D engine becomes a regression test.** A 3-D code with no object in it,
integrated over the surface, must reproduce the v1 albedo exactly. Not
approximately — the same physics computed two ways. That is the strongest
possible check on the new geometry, and it is free.

**Diffusion theory is the analytic oracle again.** The semi-infinite Green's
function with an extrapolated boundary — the one that produced the `rho/3`
table above — predicts `R(rho)` and `R(rho, t)` in closed form for a
homogeneous pack. The 3-D Monte Carlo must reproduce it wherever diffusion
theory is valid, which for snow is nearly everywhere beyond a few `l*`.

**Born approximation for the object.** For a weak, small inclusion the contrast
has a closed form in terms of the same Green's functions. It will fail for a
large strongly absorbing object — a person is not a perturbation — but it
brackets the small-object limit, and a Monte Carlo that disagrees with it
*there* is wrong.

**Publish the limits, not just the successes.** The deliverable is a
detectability surface: depth versus object size versus impurity loading versus
wavelength, with the SNR and clutter assumptions stated. Where it says
"undetectable", that is the result.

## Honest positioning

Avalanche rescue already has RECCO (866 MHz radar), transceivers (457 kHz) and
ground-penetrating radar. They win at depth, and for a hard physical reason:
dry snow is nearly transparent at radio frequencies, so those systems work
through metres. Optical is confined to tens of centimetres by the impurity
result above, and no amount of engineering moves that.

So this work should not be framed as competing with beacons. Where it stands up:

1. **Detectability-limit theory.** A rigorous, falsifiable answer to how deep
   optical detection can reach and what governs it. Publishable on its own, and
   the impurity dominance is a genuinely non-obvious finding.
2. **Very clean snow.** Antarctic and interior Greenland snow at 0.1–0.3 ng/g
   puts half a metre within reach. Different applications — buried instruments,
   crevasse bridging, firn structure.
3. **Voids rather than absorbers.** A cavity perturbs the diffusion far more
   strongly than an absorbing body of the same size, because it removes
   scattering rather than adding absorption. Contrast is much more favourable,
   and crevasse detection is a real problem.
4. **Snowpack stratigraphy.** The clutter that limits object detection is
   itself the signal for a different question. The same instrument that
   struggles to find a buried pack is well suited to profiling layer structure
   non-destructively.

Direction 3 is the one worth examining first, because the physics favours it
and nobody has to be rescued for the result to matter.

## Caveats

- Ice constants are now the real Warren & Brandt (2008) compilation, and every
  depth here was recomputed against it. Below 390 nm their `k` is a reported
  upper limit rather than a measurement, so this note quotes 400 nm.
- Grain sizes are log-normal with `sigma_g = 1.5`. A monodisperse calculation
  would put resonance spikes in these numbers; see the README.
- Impurity concentrations by region are quoted from memory and need proper
  citation before they appear in anything external.
- Diffusion-theory numbers assume a homogeneous semi-infinite pack. Real
  layering is exactly what the clutter section says will break this.
- Black carbon is treated as externally mixed, so the darkening — and hence the
  depth limits — are conservative. Internal mixing absorbs roughly twice as
  strongly per unit mass and would shorten every depth quoted here.

## Suggested sequence

1. Finish v1. The four benchmark reproductions are the credentials; without
   them no detectability number is believable.
2. Add path-length accumulation to the 1-D engine and validate against the
   time-domain diffusion solution. Cheap, and it de-risks the most valuable
   observable.
3. Build the 3-D transport with no object, and prove it reproduces v1.
4. Add geometry, then a single spherical inclusion, and check the small-object
   limit against Born.
5. Only then, the detectability surface.
