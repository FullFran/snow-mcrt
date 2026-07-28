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

## The Mie cache is a decorator over the port

The claim above — that Mie evaluation is milliseconds and the transport is
where the time goes — is true per *call* and misleading per *run*. Measured on
this repository, regenerating the four detectability figures spends 62 seconds
almost entirely inside `miepython`, and `run_albedo.py` spends 237.

The reason is not that any single evaluation is slow. It is that the same
evaluation is requested over and over. `figure_detectability_map` sweeps black
carbon loading across a fixed ice grain population, so the ice Mie table is
identical on every iteration; the impurity is the only thing that changed.
Counted directly: of the 19 578 `(m, x)` points that one script asks for,
**1 258 are distinct**. Ninety-four percent of the work was already redundant
*within a single process*, before any question of reruns.

So `adapters/cached_mie_solver.py` wraps the port rather than modifying the
solver. `CachedMieSolver` implements `MieSolver`, holds another `MieSolver`,
and memoises `efficiencies` in memory and on disk. `domain/` is untouched and
cannot tell the difference; a run drops the caching by not wrapping, and
`--no-cache` on every script does exactly that.

**The key is the exact bits of `(Re m, Im m, x)`.** Not rounded, and not a
hash over the whole array. Both choices are load-bearing:

- Exact bits mean a hit is only possible on bit-identical input, so the cache
  cannot introduce numerical error. It returns precisely what the wrapped
  solver would have returned, or it calls the wrapped solver. There is no
  third outcome, and no tolerance to argue about later. A tolerant key would
  be a second, silent error source stacked on top of the transport noise —
  in a codebase whose whole point is validating against published benchmarks
  to a stated residual, that trade is not available.
- Per-element keys mean extending a wavelength grid costs only the new points.
  A whole-array hash would discard the entire table whenever one endpoint
  moved, which is precisely what happens while a figure is being tuned.

Grids here come from `np.linspace`/`np.logspace` and `LogNormalGrainSizes`,
driven by manifest parameters, so they reproduce bit-for-bit and exact keys
hit. A grid arrived at another way simply misses, and a miss is always correct.

Measured end to end, with the CSVs compared byte for byte against an uncached
run:

| script | uncached | first cached run | rerun |
| ------ | -------- | ---------------- | ----- |
| `plot_detectability.py` | 61.7 s | 6.2 s | 2.5 s |
| `run_albedo.py` | 237.4 s | 210.1 s | 0.6 s |

The two columns measure different things. `plot_detectability.py` gains ten
times on its *first* run, because its redundancy is within the process.
`run_albedo.py` gains almost nothing there — its nine curves genuinely need
different grain populations — and everything on the rerun.

It is an optimisation and fails like one: a corrupt, truncated, or
version-mismatched file costs time and never a run. Saves merge with what is
already on disk and land through `os.replace`, so two scripts running at once
neither tear a file nor erase each other's work.

`phase_function` is deliberately not cached. An entry is an array over the
angle grid rather than three floats, and nothing in the spectral pipeline
calls it in a loop — a lot of disk for no measured time.

### Two tables, because there are two jobs

The obvious next question is whether the table behind the *committed* results
should itself be committed. It should — but not the same file.

- **`data/mie/` — frozen, committed, read-only.** It covers exactly the
  configurations that produced the CSVs in `data/reference` and the figures in
  `docs/figures`, and it is regenerated deliberately by
  `scripts/build_mie_cache.py`. `--check` reports whether it still covers
  every published run. 26 218 distinct points, 1.3 MB, seven minutes to
  rebuild from nothing.
- **`$SNOW_MCRT_CACHE_DIR` (default `~/.cache/snow-mcrt`) — mutable, local.**
  Scratch space that grows on its own as grids are widened and figures tuned.

Readers consult both and write only to the local one, so points already in the
frozen table are never copied into it.

The split is not fastidiousness. `npz` is compressed binary, so git cannot
delta it: a *mutable* cache under version control would rewrite the whole file
on every run and turn the history into a pile of blobs. A table regenerated on
purpose changes when the published results change, which is what a
reproducibility artifact should do.

The point of committing it is not the disk it saves on a rerun — the CSVs and
manifests already reproduce the published figures without any physics. It is
the person who clones the repository to run a configuration that *does not*
exist yet: a new impurity loading, a different density, another layer
thickness. The CSVs cannot help them; the table can, and saves them four
minutes of saturated CPU.

### Provenance, and why a mismatch recomputes

A committed table needs a stamp, and the bit-exact key is exactly why. Without
one, a reader on a newer `miepython` asks for a point, gets a **hit**, and
walks away with someone else's numbers believing they are their own. A cache
able to silently mask a change of solver is not a cache; it is a way to
invalidate a validation, in a repository whose entire claim is agreement with
published benchmarks to a stated residual.

So every table carries solver, solver version, NumPy version, and platform,
and a stamp that does not match is **refused and recomputed** — with a warning,
because silence there would look exactly like "the committed table did not
cover your grid", a different and much less interesting problem. This is the
same choice `OpticalConstants.m_at` makes when asked to extrapolate: raise
rather than quietly serve a plausible number.

Two consequences follow, and both are accepted rather than worked around.
`version` is part of the `MieSolver` port, because a number nobody can
attribute to a specific implementation is not evidence. And `miepython` is
pinned exactly in `pyproject.toml` rather than floated, because a committed
table attributable to "some 3.x" is not attributable at all — and under a
floating range the stamp would drift and the table would quietly stop being
used. The honest cost: on a machine whose environment differs, the committed
table buys nothing. That is the price of not lying about where a number came
from.

The Python patch version is deliberately *not* in the stamp. The series is
evaluated in NumPy, and pinning tables to an interpreter release would refuse
them after an upgrade that cannot have changed a result.

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

## Monodisperse spheres are an actively misleading idealisation

Found by looking at a figure, not by a test — which is the argument for
plotting results rather than only asserting them.

The first spectral albedo curve computed from the real Warren & Brandt
constants had a single-point notch at 676.7 nm. The co-albedo jumped from
3.0e-5 to 2.0e-4 and back, in a quantity that should vary smoothly.

It is not a bug. It is a **morphology-dependent resonance** — a
whispering-gallery mode of a perfect sphere. Resolved at fixed refractive
index, the peak sits 13.4 times above the local background over a feature only
`0.48` wide in `x`, at `x ~ 928`. Real physics, for one perfect sphere.

And pure fiction for a snowpack. Averaging over grain size destroys it:

| `sigma_g` | co-albedo at 676.7 nm |
| --------- | --------------------- |
| 1.0 (monodisperse) | 2.04e-04     |
| 1.05 (a 5% spread) | 3.27e-05     |
| 1.5 (real snow)    | 4.80e-05     |

A five percent spread — far narrower than any real snow — already restores the
smooth background. So `SnowLayer` defaults to `sigma_g = 1.5`, and reaching
monodisperse behaviour requires passing `1.0` deliberately. Defaults should be
the safe choice, not the simple one.

### The quadrature must have an even number of nodes

This one took a second pass to find, because the first fix looked like it had
worked and had not.

A symmetric quadrature with an *odd* node count puts a node exactly on the
median radius, and the log-normal gives that node the largest weight in the
distribution. When the median radius happens to sit on a resonance — as 100 um
grains do at 676.7 nm — the resonance survives the averaging that was supposed
to destroy it. Measured at that wavelength, against smooth neighbours of 4.4e-5
and 5.5e-5:

| nodes | co-albedo | |
| ----- | --------- | - |
| 16 | 4.70e-05 | even |
| **17** | **6.65e-05** | odd — median sampled |
| 32 | 4.68e-05 | even |
| **33** | **5.66e-05** | odd — median sampled |
| 48 | 4.72e-05 | even |

`LogNormalGrainSizes` therefore *refuses* an odd node count rather than
accepting it and returning a quietly contaminated answer.

### What is left, and why more nodes will not fix it

Sub-percent structure remains in the co-albedo, and it is honest to say so.

It is not a bug and it is not a quadrature artefact. Perturbing `x` by one part
in `1e8` moves `Q_abs` smoothly and monotonically — we are on the flank of a
genuinely sharp resonance, not in numerical noise. Against the geometric-optics
limit, `Q_abs / 4kx` comes out at 0.86, 0.90 and 0.84 across three size
parameters: consistent, so the magnitudes are right.

Throwing nodes at it does not help. Roughness measured over 700–760 nm: 0.0225
at 16 nodes, 0.0177 at 256. A twentyfold cost for a fifth off. The resonance
comb is dense enough that any finite quadrature lands on some of it, and the
convergence is not monotonic because different node counts hit different
resonances.

At the level it survives — a fraction of a percent in `1 - omega`, which is
around 0.005 in albedo — it is invisible in the figures and well below the
accuracy of everything else in the chain. `Q_abs` is a difference of two
numbers near 2.02 that differ by 7e-5, so roughly four and a half decimal
digits are lost to cancellation before anything else happens. That, not the
quadrature, is the floor.

Two consequences worth stating:

- Number density is fixed by mass, so it is `<r³>` that enters, not the median
  cubed. For a log-normal they differ by `exp(4.5 ln²sigma)` — a factor of 2.1
  at `sigma_g = 1.5`. Using the median would overcount grains and inflate
  extinction by that factor.
- The quadrature is **another axis of the array**, integrated in one solver
  call rather than a loop over radii. The first implementation looped, and the
  reference sweep went from seconds to tens of minutes. The same discipline
  that makes photons an array axis applies here.

The quadrature spans ±3 standard deviations, not ±4. The tail beyond carries a
thousandth of the grains and costs more than everything else combined: at
`sigma_g = 1.5` the fourth deviation reaches five times the median radius,
which for millimetre grains in the near ultraviolet means `x > 1e5`.

## A limitation this does not fix

The committed reference curves use 160 logarithmically spaced wavelengths, and
that grid **does not resolve the sharp ice absorption edges**. Across 1390 to
1447 nm the albedo of 100 um snow falls 0.386 → 0.282 → 0.137 → 0.073 in four
points. The curve there is under-resolved, and a spike detector fires on it.

That is a resolution limit, not a spurious feature, and it is distinguishable
from one: the 676.7 nm resonance had an isolated-spike score of ~60 against a
background of ~15, and after size averaging it reads 13.9 — indistinguishable
from ordinary curvature. The 1400 nm edge scores 71 because it is genuinely
that steep. Raising the point count is the fix if those bands ever matter.

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

## Which oracle, and a correction

The two-stream albedo was introduced here as "the oracle". Measuring it
against the Monte Carlo showed that framing was wrong, and the correction is
worth recording because it changes what the tests are allowed to assert.

`semi_infinite_albedo` (two-stream) is an *approximation*, and not a tight one
away from the conservative limit:

| `omega` | Monte Carlo | van de Hulst | MC error | two-stream | two-stream error |
| ------- | ----------- | ------------ | -------- | ---------- | ---------------- |
| 0.50    | 0.1474      | 0.1445       | +1.96%   | 0.1716     | **+18.71%**      |
| 0.80    | 0.3428      | 0.3403       | +0.73%   | 0.3820     | +12.23%          |
| 0.90    | 0.4781      | 0.4772       | +0.20%   | 0.5195     | +8.87%           |
| 0.95    | 0.5961      | 0.5963       | −0.03%   | 0.6345     | +6.41%           |
| 0.99    | 0.7938      | 0.7945       | −0.09%   | 0.8182     | +2.98%           |

The transport agrees with van de Hulst's fit to the exact H-function solution
to a fraction of a percent in the high-albedo regime. Two-stream is off by up
to 19%. Judging the Monte Carlo against two-stream would have meant chasing a
"bug" that was the yardstick all along.

So there are two functions and they have different jobs.
`van_de_hulst_semi_infinite_albedo` (with `similarity_scaled_albedo` for
anisotropic media) is the quantitative oracle. `semi_infinite_albedo` stays
because delta-Eddington work needs it and because it is the right tool for
reasoning, but it does not judge correctness.

The albedo and e-folding figures quoted earlier in this document are
unaffected: clean snow in the visible sits at `1 - omega ~ 4e-6`, where the
two agree to well under a tenth of a percent.

## What actually constrains the transport

Ordered by strength.

**Exact invariants** hold for any parameters and admit no tuning:

- A conservative slab returns every photon: `R + T = 1` to 1e-12, with zero
  absorbed and zero truncated.
- Without Russian roulette the energy ledger closes to float64 epsilon,
  because nothing is ever discarded.
- A purely absorbing medium reflects exactly nothing.

**Roulette conserves energy only in expectation.** Killed photons take their
weight with them; survivors are boosted by `1/p`. Measured residuals are
`−1.4e-6` and `+1.9e-6` — small, and *sign-changing between runs*, which is
the signature of an unbiased estimator rather than a leak. With roulette off,
the same runs close to `1e-16`. The test suite asserts both separately;
demanding exactness with roulette on would be demanding that an estimator be a
conserved quantity.

**Similarity scaling** cross-checks the angular sampling: a run with
`(omega, g)` must land on the same albedo as one with the equivalent isotropic
`omega* = omega(1-g)/(1-omega g)`. Measured agreement is 0.4–0.6%, which is
the accuracy of similarity theory itself. If the deflection rotation were
wrong, forward scattering would not map onto its isotropic equivalent at all.

**Truncation is reported, never absorbed.** `TransportResult.truncated` carries
weight still in flight when `max_scatters` runs out. Folding it into
absorption would turn a convergence failure into a plausible albedo with no
indication anything went wrong. A test starves a run to five scattering orders
and asserts the shortfall surfaces there.

## Why the GPU is not optional after all

The cost of this problem is physics, not implementation. Light really does
scatter tens of thousands of times inside snow before it leaves, and the mean
number of scattering orders scales as `1/(1-omega)`.

End-to-end runs, 20 000 photons on one CPU core, full pipeline from refractive
index through Mie and mixing to transport:

| case                    | `1 − omega` | MC albedo | oracle | scatters | time  |
| ----------------------- | ----------- | --------- | ------ | -------- | ----- |
| 1 mm, 1300 nm, clean    | 9.3e-02     | 0.1288    | 0.1308 | 189      | 0.3 s |
| 100 um, 1300 nm, clean  | 1.1e-02     | 0.4900    | 0.4945 | 1 520    | 2.1 s |
| 100 um, 500 nm, 1000 ng/g BC | 4.4e-04 | 0.8591 | 0.8649 | 31 058   | 40 s  |
| 100 um, 500 nm, 100 ng/g BC  | 4.8e-05 | 0.9500 | 0.9531 | 200 000* | 260 s |

`*` hit the scattering cap.

Clean snow in the visible is `1 - omega ~ 4e-6`, another two orders of
magnitude beyond the last row. That is the headline case — research question 1
— and it is out of reach of a single CPU core at useful photon counts. The
array-backend port was justified earlier as a correctness device; this table is
the performance half of the argument, and it is measured rather than asserted.

Note also that every case sits slightly *below* the oracle, by 0.3–1.5%. That
is the direction similarity theory is known to err, and it is consistent
across four independent cases rather than scattered — evidence the residual is
the approximation, not the transport.

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
- `test_cached_mie_solver.py` — the caching decorator, checked against a
  counting stub rather than the real solver. The question is not whether Mie
  theory is right (`test_mie.py` owns that) but whether the wrapper returned
  exactly what the wrapped solver would have and skipped calling it twice. A
  stub makes both observable; the real solver makes neither.
- `test_backends.py` — the port contract, checked against both adapters. The
  CuPy *numerical* tests skip without CUDA; the *contract* tests, including
  that `CupyBackend` refuses to construct when CuPy is absent, run everywhere.
  Those are the ones that catch a port drifting away from its adapters.
