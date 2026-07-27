# Ice optical constants — provenance

## `warren_brandt_2008.dat`

**Dataset:** Optical constants of ice from the ultraviolet to the microwave,
revised compilation.

**Citation:** Warren, S. G., & Brandt, R. E. (2008). *Journal of Geophysical
Research: Atmospheres*, 113, D14220. DOI:
[10.1029/2007JD009744](https://doi.org/10.1029/2007JD009744)

**Retrieved from:** `https://atmos.washington.edu/ice_optical_constants/IOP_2008_ASCIItable.dat`

**Retrieved on:** 2026-07-27

**Format:** three whitespace-separated columns, no header.

| column | quantity                         | units      |
| ------ | -------------------------------- | ---------- |
| 1      | wavelength                       | micrometres |
| 2      | `n`, real refractive index       | —          |
| 3      | `k`, imaginary refractive index  | —          |

486 rows, ascending, spanning **44.3 nm to 2 m** — ultraviolet through the
microwave. Load with `wavelength_scale_to_nm=1000.0`.

## Two things to know before using it

**The imaginary part has a reported floor of `k = 2.0e-11`.** It sits flat at
that value from 202 nm to 390 nm. This is not a measured minimum, it is an
*upper limit*: in that band ice absorbs too weakly for the available
measurements to resolve, and the compilation reports the bound rather than a
value.

Consequences, both of which matter for this project:

- Any absorption or penetration depth computed between 202 and 390 nm is a
  **bound, not a prediction**. True penetration there could be deeper.
- The deepest wavelength with a genuinely measured value is around **400 nm**,
  where `k = 2.365e-11`. That is the figure to quote for maximum penetration
  in pure ice.

**Convention.** The file gives `k` as a positive number. The domain convention
is `m = n + ik` with `k >= 0`, so it loads directly with no sign change. The
conversion to `miepython`'s `m = n - ik` happens in that adapter, not here.

## Spot values

Interpolated from the table, for orientation:

| wavelength | `n`    | `k`       |
| ---------- | ------ | --------- |
| 400 nm     | 1.3194 | 2.365e-11 |
| 450 nm     | 1.3157 | 9.239e-11 |
| 500 nm     | 1.3130 | 5.889e-10 |
| 1300 nm    | 1.2961 | 1.320e-05 |

Note the range: `k` climbs by nearly six orders of magnitude from 400 nm to
1300 nm. That span is the reason snow is brilliant white to the eye and dark
in the near infrared, and it is why `OpticalConstants.m_at` refuses to
extrapolate rather than clamping at an endpoint.
