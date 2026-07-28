"""What happens where the index of refraction changes.

The plane-parallel transport in :mod:`~snow_mcrt.domain.transport` has no
interface at all: a photon reaching ``z = 0`` has left, and that is the end of
it. For a spectral albedo against a semi-infinite pack that is defensible,
because the analytic oracle it is checked against makes the same assumption.

It stops being defensible the moment the geometry is three-dimensional and the
question is a *spatially resolved* measurement. Snow is ``n = 1.31``, and while
that is a weak interface head-on -- 1.8% at normal incidence -- the diffuse
flux arriving at the surface from inside arrives at every angle, and a large
part of it arrives beyond the critical angle of 49.8 degrees, where reflection
is **total**. Integrated over a Lambertian interior distribution, roughly half
the flux reaching the boundary is turned back into the pack. That is why
diffusion theory needs an extrapolated boundary sitting well outside the
surface, and it is why a 3-D engine without Fresnel would disagree with the
diffusion solution by a factor that looks like a bug and is not.

Conventions, fixed here so nothing downstream has to think about them:

- Directions are unit row vectors, shape ``(n, 3)``.
- ``normal`` points *against* the incoming ray, so ``cos_i = -dot(d, n) >= 0``.
  Orienting the normal is the caller's job; the geometry module does it at the
  point where it already knows which side was hit.
- ``n1`` is the index the photon is leaving, ``n2`` the one it would enter.

Reflection or transmission is decided by a **single random draw**, not by
splitting the photon's weight between two outcomes. Both are unbiased. The
draw is chosen because it keeps one photon in one array slot: splitting would
grow the population at every interface, and a fixed-width array is the whole
reason this engine vectorises.

References: Born & Wolf for the Fresnel coefficients; Wang, Jacques & Zheng
(1995), *MCML -- Monte Carlo modeling of light transport in multi-layered
tissues*, for the vector forms used in a Monte Carlo stepper.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def critical_angle(n1: float, n2: float) -> float:
    """Angle past which nothing is transmitted, radians.

    ``nan`` when ``n2 >= n1``: going into a denser medium there is no total
    internal reflection, and a caller comparing against ``nan`` gets ``False``,
    which is the correct answer to "am I past the critical angle".
    """
    if n2 >= n1:
        return float("nan")
    return float(np.arcsin(n2 / n1))


def fresnel_reflectance(cos_incident: Any, n1: float, n2: float, xp: Any = np) -> Any:
    """Unpolarised reflectance at the interface.

    ``R = (|r_s|^2 + |r_p|^2) / 2`` -- the average of the two polarisations,
    which is what an unpolarised photon sees. Beyond the critical angle the
    transmitted direction does not exist and the result is exactly ``1.0``, not
    a large number close to it: there is no transmitted branch to take.

    Args:
        cos_incident: ``cos`` of the angle to the normal, in ``[0, 1]``.
        n1: Index the photon is in.
        n2: Index on the other side.
        xp: Array namespace, so this runs unchanged on a GPU backend.

    Returns:
        Reflectance in ``[0, 1]``, broadcast to the shape of ``cos_incident``.
    """
    cos_i = xp.clip(cos_incident, 0.0, 1.0)
    if n1 == n2:
        return xp.zeros_like(cos_i * 1.0)

    sin_t_sq = (n1 / n2) ** 2 * (1.0 - cos_i**2)
    total = sin_t_sq >= 1.0
    # Evaluate the transmitted cosine on a clamped argument so the total
    # internal reflection branch cannot produce a nan that then propagates
    # through xp.where -- an unselected branch is still computed.
    cos_t = xp.sqrt(xp.maximum(1.0 - sin_t_sq, 0.0))

    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
    reflectance = 0.5 * (r_s**2 + r_p**2)
    return xp.where(total, xp.ones_like(reflectance), reflectance)


def specular_reflect(direction: Any, normal: Any, xp: Any = np) -> Any:
    """Mirror ``direction`` about the plane with the given ``normal``.

    ``d + 2 (d . n_in) n`` with ``n`` oriented against the ray, which flips the
    component along the normal and leaves the tangential part untouched.
    """
    cos_i = -xp.sum(direction * normal, axis=-1, keepdims=True)
    return direction + 2.0 * cos_i * normal


def refract(direction: Any, normal: Any, n1: float, n2: float, xp: Any = np) -> Any:
    """Snell's law in vector form.

    Undefined beyond the critical angle; the caller must have reflected
    instead. The square root is clamped at zero so a caller that gets this
    wrong receives a grazing direction rather than a ``nan`` that spreads
    silently through the rest of the run.
    """
    ratio = n1 / n2
    cos_i = -xp.sum(direction * normal, axis=-1, keepdims=True)
    cos_t = xp.sqrt(xp.maximum(1.0 - ratio**2 * (1.0 - cos_i**2), 0.0))
    return ratio * direction + (ratio * cos_i - cos_t) * normal
