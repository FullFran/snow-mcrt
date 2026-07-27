"""Closed-form radiative transfer -- the oracle the Monte Carlo is checked against.

A Monte Carlo transport code produces smooth, confident-looking albedo curves
whether or not it is correct. Without an independent calculation there is
nothing to falsify it with, so these approximations are domain knowledge, not
test scaffolding: they are what the transport must reproduce in the regimes
where they hold, and they also answer research question 3 directly.

The two-stream results here are asymptotic -- semi-infinite, homogeneous,
diffuse illumination. That is exactly the regime deep clean snow sits in, and
it is where a transport bug has nowhere to hide.

References:
    Wiscombe & Warren (1980), J. Atmos. Sci. 37, 2712.
    Joseph, Wiscombe & Weinman (1976), J. Atmos. Sci. 33, 2452 (delta-Eddington).
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
