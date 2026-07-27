"""Closed-form radiative transfer -- the oracle the Monte Carlo is checked against.

A Monte Carlo transport code produces smooth, confident-looking albedo curves
whether or not it is correct. Without an independent calculation there is
nothing to falsify it with, so these approximations are domain knowledge, not
test scaffolding: they are what the transport must reproduce in the regimes
where they hold, and they also answer research question 3 directly.

**Two of them, and they are not interchangeable.** This matters enough to say
before the code:

- :func:`semi_infinite_albedo` is *two-stream*. It is an approximation and its
  error is large away from ``omega -> 1``: measured against an accurate
  reference it runs 19% high at ``omega = 0.5``, 8.9% high at 0.9, 3.0% high
  at 0.99. It converges only in the conservative limit. It is the right tool
  for understanding and for delta-Eddington work, and the wrong tool for
  judging whether a Monte Carlo run is correct.
- :func:`van_de_hulst_semi_infinite_albedo` is accurate to about 1% across the
  whole range for isotropic scattering. **This is the quantitative oracle.**

Clean snow in the visible sits at ``1 - omega ~ 4e-6``, deep in the regime
where the two agree, so the albedo figures either produces there are sound.
The distinction bites in the middle of the range, which is exactly where a
transport bug would show up first.

References:
    Wiscombe & Warren (1980), J. Atmos. Sci. 37, 2712.
    Joseph, Wiscombe & Weinman (1976), J. Atmos. Sci. 33, 2452 (delta-Eddington).
    van de Hulst, H. C. (1974), Astron. Astrophys. 35, 209.
"""

from __future__ import annotations

import numpy as np


def delta_eddington_scaling(
    omega: np.ndarray | float, g: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale away the forward diffraction peak.

    A large sphere throws roughly half its scattered energy into a peak
    narrower than any practical angular grid. Delta-Eddington treats that
    fraction ``f = g^2`` as unscattered and rescales what remains:

    .. math::
        g' = \\frac{g}{1 + g},\\quad
        \\omega' = \\frac{(1 - f)\\omega}{1 - f\\omega}

    Optical depth scales as ``tau' = (1 - f*omega) tau`` -- applied by the
    caller, since it depends on the geometry rather than on the scattering.

    Returns:
        Scaled ``(omega, g)``.
    """
    omega = np.asarray(omega, dtype=float)
    g = np.asarray(g, dtype=float)
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("single-scattering albedo must lie in [0, 1]")
    if np.any(np.abs(g) >= 1):
        raise ValueError("asymmetry parameter must lie in (-1, 1)")
    f = g**2
    return (1.0 - f) * omega / (1.0 - f * omega), g / (1.0 + g)


def similarity_parameter(
    omega: np.ndarray | float, g: np.ndarray | float
) -> np.ndarray:
    """The single number deep snow optics actually depends on.

    .. math::
        s = \\sqrt{\\frac{1 - \\omega}{1 - \\omega g}}

    Two snowpacks with different ``omega`` and ``g`` but equal ``s`` are
    optically indistinguishable at depth. This is why grain shape can be
    folded into an effective radius at all, and why v1 gets away with
    spheres.
    """
    omega = np.asarray(omega, dtype=float)
    g = np.asarray(g, dtype=float)
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("single-scattering albedo must lie in [0, 1]")
    return np.sqrt((1.0 - omega) / (1.0 - omega * g))


def semi_infinite_albedo(
    omega: np.ndarray | float, g: np.ndarray | float, delta_scaled: bool = True
) -> np.ndarray:
    """Diffuse albedo of a semi-infinite homogeneous snowpack.

    .. math::
        \\alpha = \\frac{1 - s}{1 + s}

    Args:
        omega: Single-scattering albedo.
        g: Asymmetry parameter.
        delta_scaled: Apply delta-Eddington scaling first. Leave this on for
            snow; the raw asymmetry parameter of an ice grain is ~0.89 and
            two-stream handles a peak that sharp badly.

    Returns:
        Albedo in ``[0, 1]``. A conservative medium (``omega = 1``) returns
        exactly 1, as it must: with nothing absorbing, everything that goes in
        comes back out.
    """
    if delta_scaled:
        omega, g = delta_eddington_scaling(omega, g)
    s = similarity_parameter(omega, g)
    return (1.0 - s) / (1.0 + s)


def van_de_hulst_semi_infinite_albedo(omega: np.ndarray | float) -> np.ndarray:
    """Accurate semi-infinite albedo for **isotropic** scattering.

    .. math::
        \\alpha = \\frac{(1 - s)(1 - 0.139 s)}{1 + 1.17 s},
        \\quad s = \\sqrt{1 - \\omega}

    A fit to the exact H-function solution, good to about 1% over the whole
    range of ``omega``. This is the quantitative oracle the Monte Carlo is
    judged against, because :func:`semi_infinite_albedo` is not accurate
    enough to serve: at ``omega = 0.9`` the two differ by 8.9%, which is far
    larger than any transport bug worth catching.

    Restricted to isotropic scattering by construction. Anisotropic cases are
    reached through similarity scaling -- ``omega* = omega(1-g)/(1-omega g)``
    -- which is itself approximate at the half-percent level.

    Args:
        omega: Single-scattering albedo in ``[0, 1]``.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("single-scattering albedo must lie in [0, 1]")
    s = np.sqrt(1.0 - omega)
    return (1.0 - s) * (1.0 - 0.139 * s) / (1.0 + 1.17 * s)


def similarity_scaled_albedo(
    omega: np.ndarray | float, g: np.ndarray | float
) -> np.ndarray:
    """Anisotropic semi-infinite albedo via similarity scaling.

    Maps ``(omega, g)`` onto the isotropic problem with
    ``omega* = omega(1-g)/(1-omega g)`` and evaluates
    :func:`van_de_hulst_semi_infinite_albedo` there.

    Accurate to roughly half a percent for the strongly forward-scattering
    media snow presents, which is good enough to catch a transport bug and not
    good enough to be called exact.
    """
    omega = np.asarray(omega, dtype=float)
    g = np.asarray(g, dtype=float)
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("single-scattering albedo must lie in [0, 1]")
    if np.any(np.abs(g) >= 1):
        raise ValueError("asymmetry parameter must lie in (-1, 1)")
    return van_de_hulst_semi_infinite_albedo(
        omega * (1.0 - g) / (1.0 - omega * g)
    )


def asymptotic_extinction_coefficient(
    omega: np.ndarray | float,
    g: np.ndarray | float,
    extinction_coefficient: np.ndarray | float,
    delta_scaled: bool = True,
) -> np.ndarray:
    """Decay rate of the diffuse flux with depth, in inverse metres.

    .. math::
        k_e = \\beta \\sqrt{3 (1 - \\omega)(1 - \\omega g)}

    Deep in the pack the radiance field reaches an asymptotic regime where its
    shape stops changing and only its amplitude decays, at this rate. It is
    much slower than the extinction coefficient ``beta``: photons scatter
    thousands of times before being absorbed, so light penetrates far deeper
    than a single mean free path.

    Args:
        omega: Single-scattering albedo.
        g: Asymmetry parameter.
        extinction_coefficient: Bulk extinction ``beta``, m^-1.
        delta_scaled: Apply delta-Eddington scaling first.
    """
    beta = np.asarray(extinction_coefficient, dtype=float)
    if np.any(beta < 0):
        raise ValueError("extinction coefficient must be non-negative")
    if delta_scaled:
        omega, g = delta_eddington_scaling(omega, g)
    omega = np.asarray(omega, dtype=float)
    g = np.asarray(g, dtype=float)
    return beta * np.sqrt(3.0 * (1.0 - omega) * (1.0 - omega * g))


def e_folding_depth(
    omega: np.ndarray | float,
    g: np.ndarray | float,
    extinction_coefficient: np.ndarray | float,
    delta_scaled: bool = True,
) -> np.ndarray:
    """Depth over which the diffuse flux falls by a factor of ``e``, in metres.

    The reciprocal of :func:`asymptotic_extinction_coefficient`, and the
    quantity measured in the field -- research question 3 compares against it
    directly.

    Returns:
        Depth in metres, ``inf`` for a conservative medium where the flux
        never decays.
    """
    k = asymptotic_extinction_coefficient(
        omega, g, extinction_coefficient, delta_scaled=delta_scaled
    )
    with np.errstate(divide="ignore"):
        return np.where(k > 0, 1.0 / np.where(k > 0, k, 1.0), np.inf)
