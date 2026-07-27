# Architecture

Why this repository is laid out the way it is. The short version: a Monte Carlo
simulation that has to run on two very different devices, be validated against
closed-form radiative transfer, and produce a reproducible artefact has three
distinct concerns, and keeping them apart is what makes the bugs findable.

```
src/snow_mcrt/
  domain/        optics, Mie properties, transport, observables, analytic solutions
  ports/         the protocols the physics depends on: array backend, Mie solver, optical data
  adapters/      NumPy (oracle) and CuPy (production); miepython; tabulated datasets
  application/   one use case per research question
  infra/         CSV and manifest writers
scripts/         headless entry points: run a sweep, render figures
tests/           validation against ground truth
data/reference/  committed results, so figures reproduce without a GPU
notebooks/       one per benchmark reproduction
```

## The three ports

The simulation core never imports NumPy, CuPy, or `miepython`. It receives
protocols and works through them.

This is not architecture for its own sake. Each port buys something concrete.

**`backend.py` — the array namespace.** NumPy and CuPy expose nearly identical
array APIs, so the port only covers what genuinely differs: random number
generation, device transfer, and module identity. Everything in `domain/` is
written once and runs on both.

It also closes off a specific performance bug before it can be written. A
photon-transport loop draws random numbers every step for every live photon.
Allocating them on the host and copying them to the device is the dominant cost
in a naive GPU Monte Carlo, and it is invisible in a profile that only times
the kernel. `random_uniform` is part of the port precisely so that each backend
is *required* to allocate on its own device.

**`mie_solver.py` — the series evaluation.** Mie theory is domain physics, but
the recursion over Riccati-Bessel functions is a solved numerical problem.
The port draws the line: `domain/mie.py` owns what the efficiencies mean and
how they combine into snowpack optical properties; the recursion lives in an
adapter. A second solver can be dropped in and cross-checked, which is the only
way to distinguish a subtle convergence failure at large size parameters from a
real physical result. Snow needs `x ~ 1.6e4` for a 1 mm grain at 400 nm, which
is well into the range where naive upward recurrence loses all its digits.

**`optical_data.py` — where the refractive index comes from.** A tabulated
file, a published fit, or a synthetic dataset built for a test. Keeping this
behind a port is what lets the physics tests run on analytic data with known
answers instead of on a multi-megabyte file whose correctness is itself an
open question.

## The NumPy adapter is the oracle

It is not a fallback for machines without a GPU. Every physics test runs
against it, because on small photon counts its output can be compared to
closed-form two-stream solutions and to published benchmark tables. If CuPy
ever disagrees with NumPy, CuPy is wrong. Without a second independent
implementation there is nothing to cross-check against.

## Where the GPU actually pays

In the transport loop, and only there.

Mie evaluation over a wavelength grid is thousands of values. It runs on the
host in milliseconds, and moving it to a device would be motion rather than
speed. The photon loop is different: millions of photons advancing together,
each drawing free paths and scattering angles every step.

That is also what "vectorized" means here, and it is a change of model rather
than a faster loop. **Photons are an axis of the array**, carried with an
alive-mask, exactly as replicas are an axis in a parallel-tempering lattice
simulation. A single photon traced to completion and then the next is a
different program, not a slower one.

## Conventions fixed once, at the boundary

Two of them, because both are silent when wrong.

**Units.** Wavelengths are nanometres at every public boundary, because that is
how the literature and the datasets quote them. Everything else is metres:
grain radii, absorption coefficients, optical depths, geometry. A result in
inverse nanometres is off by a factor of 1e9 and still plots as a smooth curve.

**Sign.** The complex refractive index is `m = n + ik` with `k >= 0`, the
Bohren & Huffman convention. `miepython` uses `m = n - ik`. The conjugation
happens in `adapters/miepython_solver.py` and nowhere else. Passing the domain
convention straight through does not raise — it models a medium with *gain*,
and `Q_sca` merely creeps above `Q_ext` by a part in ten thousand. Nothing in
an albedo plot would ever show it. `TestSignConvention` pins it.

## Why Mie is validated before any photon exists

Everything downstream is a function of three numbers: the single-scattering
albedo `omega`, the asymmetry parameter `g`, and the bulk extinction
coefficient. Get them wrong and the transport still produces smooth,
plausible-looking albedo curves — and there is then no way to tell which of the
two layers failed.

So the single-scattering layer is validated on its own, against limits that are
closed-form:

- **Rayleigh** (`x << 1`): `Q_sca = (8/3) x^4 |(m²-1)/(m²+2)|²`, matched to a
  relative tolerance of 1e-4, and the `x^4` slope recovered from a log-log fit.
- **Geometric** (`x >> 1`): `Q_ext -> 2`, the extinction paradox.
- **Conservation**: `Q_sca <= Q_ext` across six decades of size parameter.

Between those two ends the series does work no closed form covers, which is
where the published benchmark tables take over.

## Values a reader should recognise

Produced by the current code, not copied from a paper:

| case                 | x       | Q_ext  | g      | 1 - omega | mfp (mm) |
| -------------------- | ------- | ------ | ------ | --------- | -------- |
| clean 100 um, 500 nm | 1256.6  | 2.0158 | 0.8887 | 4.37e-06  | 0.202    |
| clean 50 um, 500 nm  | 628.3   | 2.0146 | 0.8881 | 2.13e-06  | 0.101    |
| clean 1 mm, 500 nm   | 12566.4 | 2.0036 | 0.8922 | 4.22e-05  | 2.034    |
| clean 100 um, 1300nm | 483.3   | 2.0447 | 0.8935 | 1.06e-02  | 0.199    |
| clean 1 mm, 1300 nm  | 4833.2  | 2.0054 | 0.9099 | 9.33e-02  | 2.032    |

Mean free paths assume 300 kg/m³. Three things to read off it: `g ~ 0.89` is
the canonical value for ice spheres; `1 - omega` scales linearly with grain
radius; and it climbs three orders of magnitude from the visible to 1300 nm.
That last row is the grain-size signal the near-infrared benchmarks exist to
measure, and it is why visible albedo is set by trace absorbers instead.

## The forward-peak trap

A tabulated Mie phase function for snow-sized grains cannot use a uniformly
spaced `mu` grid. This is not a refinement question, it is a correctness one,
and it is worth stating plainly because the failure is loud once measured and
completely silent in use.

A sphere of size parameter `x` concentrates most of its scattered energy into a
diffraction peak of angular width around `1/x`. A 100 um grain in the visible
is `x ~ 1.3e3`, so the peak spans microradians. In `mu` it is narrower still,
because `1 - mu ~ theta² / 2` puts it within `1e-7` of unity.

A uniform grid of 20 001 points steps straight over it. Measured:

| grid                       | `integral p dmu` | `<cos>` | `g` from the series |
| -------------------------- | ---------------- | ------- | ------------------- |
| uniform, 20 001 points     | 20.41            | 20.30   | 0.8892              |
| log-spaced angle, `n=4000` | 1.006            | 0.8925  | 0.8892              |

The uniform grid produces a "phase function" that integrates to twenty and has
a mean cosine above one — not a physical quantity at all. Nothing raises at the
point of use. A transport run would simply return wrong albedos.

`forward_peaked_mu_grid` spaces angles logarithmically from `1e-7` rad to `pi`.
That lower bound is a hard float64 floor rather than a tuning choice: below it,
`1 - cos(theta)` falls under the representable spacing at unity and
neighbouring angles collapse onto the same cosine. Collapsed points are
dropped. `TabulatedPhaseFunction` then validates normalisation on
construction, so a bad grid fails at the boundary instead of downstream.

`test_a_uniform_grid_silently_destroys_the_phase_function` pins it. If that
test ever stops raising, the guard has stopped working.

## The analytic oracle

`domain/analytic.py` is the independent calculation the Monte Carlo will be
checked against. It is domain knowledge rather than test scaffolding: research
question 3 compares e-folding depth against measurement, and these are the
closed forms that produce it.

Deep clean snow sits in the asymptotic regime — semi-infinite, homogeneous,
diffuse illumination — which is exactly where two-stream theory is reliable and
where a transport bug has nowhere to hide:

- `similarity_parameter` — `s = sqrt((1-omega)/(1-omega*g))`. Two snowpacks
  with different `omega` and `g` but equal `s` are optically
  indistinguishable at depth. This is why grain shape can be folded into an
  effective radius, and why v1 gets away with spheres.
- `semi_infinite_albedo` — `(1-s)/(1+s)`, delta-Eddington scaled by default.
- `e_folding_depth` — from `k_e = beta sqrt(3(1-omega)(1-omega*g))`.

Produced by the current code, at 300 kg/m³:

| case                  | albedo | e-folding depth |
| --------------------- | ------ | --------------- |
| clean 50 um, 500 nm   | 0.9913 | 2.53 cm         |
| clean 100 um, 500 nm  | 0.9875 | 3.52 cm         |
| clean 1 mm, 500 nm    | 0.9612 | 11.23 cm        |
| clean 50 um, 1300 nm  | 0.6431 | 0.05 cm         |
| clean 100 um, 1300 nm | 0.5365 | 0.07 cm         |
| clean 1 mm, 1300 nm   | 0.1560 | 0.23 cm         |

Two things to read off it. Visible albedo moves by 3 points across a twentyfold
change in grain radius; near-infrared albedo moves by 49. That ratio is why the
two bands answer different questions — near-infrared is the grain-size
diagnostic, visible is the impurity one. And light penetrates centimetres in
the visible while stopping within a millimetre at 1300 nm, which is the same
fact seen from the other side.

Note also how far the e-folding depth exceeds a single mean free path: 3.5 cm
against 0.2 mm, a factor of 175. Photons scatter thousands of times before
being absorbed. Any transport implementation that terminates them too eagerly
will fail this comparison badly, which is the point.

## Testing strategy

Tests are organised by what they compare against, not by which module they
touch:

- `test_optics.py` — conventions: units, interpolation, and the refusal to
  extrapolate. Ice absorption varies over ten orders of magnitude across the
  solar spectrum, so a clamped endpoint is not a small error, it is a
  different material. `m_at` raises instead.
- `test_mie.py` — ground truth: the Rayleigh and geometric limits, energy
  conservation, the sign convention, and the snow regime.
- `test_phase.py` — normalisation and sampling. The Henyey-Greenstein sampler
  is checked against its own closed-form CDF in units of Poisson error, since
  the sparsest bin's one-sigma spread is already 1.4% and a flat relative
  tolerance would be either failing or vacuous depending on the bin. The Mie
  table's mean cosine is cross-checked against the `g` from the series — two
  entirely different code paths through the same solver.
- `test_analytic.py` — limits the oracle must satisfy (a conservative medium
  reflects everything; a purely absorbing one reflects nothing) and values a
  snow-optics reader recognises.
- `test_backends.py` — the port contract, checked against both adapters. The
  CuPy *numerical* tests skip without CUDA; the *contract* tests, including
  that `CupyBackend` refuses to construct when CuPy is absent, run everywhere.
  Those are the ones that catch a port drifting away from its adapters.
