# Mie table — provenance

## `mie-efficiencies-miepython-v<N>.npz`

**What it is:** memoised Mie efficiencies `(Q_ext, Q_sca, g)` for the exact
`(m, x)` points behind the committed results — the curves in `data/reference`,
the cross-validation in `data/validation`, and the figures in `docs/figures`.

**Not a cache that happens to be in the repository.** The mutable working cache
lives in `$SNOW_MCRT_CACHE_DIR` (default `~/.cache/snow-mcrt`) and is never
committed: `npz` is compressed binary that git cannot delta, so a table that
grew on every run would rewrite itself into the history as a pile of blobs.
This file changes only when the published results do.

**Why commit it at all:** the CSVs and manifests already reproduce the
published figures without re-running any physics. This table is for the case
they cannot serve — someone who clones the repository to run a configuration
that does not exist yet, at the same wavelengths and grain sizes. It saves them
about four minutes of saturated CPU.

**Regenerate:**

```bash
python scripts/build_mie_cache.py            # rebuild
python scripts/build_mie_cache.py --check    # verify coverage, write nothing
```

Never regenerated as a side effect of a run. Run scripts read it and write only
to the local cache.

## Format

An `npz` of six parallel `float64` columns plus a `provenance` field:

| field | meaning |
| ----- | ------- |
| `m_real`, `m_imag` | complex refractive index `n + ik`, `k >= 0` |
| `x` | size parameter `2*pi*r*n_medium/lambda` |
| `q_ext`, `q_sca` | extinction and scattering efficiencies |
| `g` | asymmetry parameter `<cos(theta)>` |
| `provenance` | JSON: format version, solver, solver version, NumPy, system, machine |

The lookup key is the **exact bits** of `(Re m, Im m, x)`. A hit is therefore
only possible on bit-identical input, so reuse cannot perturb a result: the
reader returns precisely what the solver would have returned, or it calls the
solver.

## Provenance is verified, and a mismatch recomputes

That bit-exactness is also the hazard. Without a stamp, a reader running a
different `miepython` would ask for a point, get a **hit**, and walk away with
numbers from another release believing they were its own — a table able to
silently mask a change of solver is a way to invalidate a validation.

So the stamp is compared on load, and one that does not match is refused with a
warning and the points are recomputed. This is the same stance
`OpticalConstants.m_at` takes when asked to extrapolate: raise rather than
quietly serve a plausible number.

Consequences, both accepted rather than worked around:

- `miepython` is pinned exactly in `pyproject.toml`. A table attributable to
  "some 3.x" is not attributable, and under a floating range the stamp would
  drift and this file would quietly stop being used.
- On a machine whose environment differs, this table buys nothing. That is the
  price of not lying about where a number came from.

The Python patch version is deliberately absent from the stamp: the series is
evaluated in NumPy, and pinning to an interpreter release would refuse the
table after an upgrade that cannot have changed a result.
