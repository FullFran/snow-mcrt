"""Single-scattering properties of spherical grains.

This module owns what the Mie efficiencies *mean* and how they combine into the
three numbers a radiative-transfer calculation actually consumes:

- the single-scattering albedo ``omega = Q_sca / Q_ext``
- the asymmetry parameter ``g = <cos(theta)>``
- the extinction coefficient of the bulk medium

Everything downstream -- two-stream, delta-Eddington, and the Monte Carlo
transport itself -- is a function of those three. Get them wrong and the
transport will still produce smooth, plausible-looking albedo curves, which is
precisely why they are validated first and on their own.

The Mie series evaluation lives behind :class:`~snow_mcrt.ports.mie_solver.MieSolver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snow_mcrt.domain.optics import size_parameter
from snow_mcrt.ports.mie_solver import MieSolver

# Ice density at 0 degrees C, kg/m^3 (Warren 1982 and the snow-optics canon).
ICE_DENSITY = 917.0


@dataclass(frozen=True)
class MieProperties:
    """Single-scattering properties of one grain population.

    All arrays share a common broadcast shape -- typically ``(n_wavelengths,)``
    or ``(n_wavelengths, n_radii)``.

    Args:
        x: Size parameter.
        q_ext: Extinction efficiency.
        q_sca: Scattering efficiency.
        g: Asymmetry parameter ``<cos(theta)>``, in ``[-1, 1]``.
        radius_m: Grain radius in metres, broadcastable against the rest.
    """

    x: np.ndarray
    q_ext: np.ndarray
    q_sca: np.ndarray
    g: np.ndarray
    radius_m: np.ndarray

    def __post_init__(self) -> None:
        if np.any(self.q_ext < 0) or np.any(self.q_sca < 0):
            raise ValueError("efficiencies must be non-negative")
        if np.any(self.q_sca > self.q_ext * (1.0 + 1e-9)):
            raise ValueError(
                "Q_sca exceeds Q_ext, which would mean negative absorption"
            )
        if np.any(np.abs(self.g) > 1.0 + 1e-9):
            raise ValueError("asymmetry parameter must lie in [-1, 1]")

    @property
    def q_abs(self) -> np.ndarray:
        """Absorption efficiency, ``Q_ext - Q_sca``."""
        return self.q_ext - self.q_sca

    @property
    def single_scattering_albedo(self) -> np.ndarray:
        """Probability that an extinction event is a scattering event.

        This is the quantity the Monte Carlo loop samples against, and for
        clean snow in the visible it sits within about ``1e-7`` of unity. That
        near-degeneracy is the whole reason visible snow albedo is so
        sensitive to trace absorbers: a few nanograms per gram of black carbon
        moves ``1 - omega`` by orders of magnitude while ``omega`` itself
        barely stirs.
        """
        return self.q_sca / self.q_ext

    @property
    def cross_section_ext(self) -> np.ndarray:
        """Extinction cross-section per grain, m^2."""
        return self.q_ext * np.pi * self.radius_m**2

    @property
    def cross_section_sca(self) -> np.ndarray:
        """Scattering cross-section per grain, m^2."""
        return self.q_sca * np.pi * self.radius_m**2

    def extinction_coefficient(self, number_density: np.ndarray | float) -> np.ndarray:
        """Bulk extinction coefficient in inverse metres.

        Args:
            number_density: Grains per cubic metre.
        """
        return np.asarray(number_density, dtype=float) * self.cross_section_ext

    def extinction_coefficient_from_density(
        self, snow_density: np.ndarray | float
    ) -> np.ndarray:
        """Bulk extinction coefficient from bulk snow density, m^-1.

        Converts a snowpack density in kg/m^3 into the number density of
        equal-sized spheres it implies, then extincts with it. Valid while the
        grains are treated as an independent, dilute population -- the same
        assumption the rest of v1 rests on.
        """
        rho = np.asarray(snow_density, dtype=float)
        if np.any(rho < 0):
            raise ValueError("snow density must be non-negative")
        if np.any(rho > ICE_DENSITY):
            raise ValueError(
                f"snow density {rho.max()} kg/m^3 exceeds solid ice "
                f"({ICE_DENSITY} kg/m^3)"
            )
        grain_mass = ICE_DENSITY * (4.0 / 3.0) * np.pi * self.radius_m**3
        return (rho / grain_mass) * self.cross_section_ext


def compute_mie_properties(
    solver: MieSolver,
    m_particle: np.ndarray | complex,
    radius_m: np.ndarray | float,
    wavelength_nm: np.ndarray | float,
    n_medium: float = 1.0,
) -> MieProperties:
    """Evaluate single-scattering properties for a grain population.

    Args:
        solver: Mie series implementation.
        m_particle: Complex index of the grain, ``n + ik``, ``k >= 0``.
        radius_m: Grain radius in metres.
        wavelength_nm: Vacuum wavelength in nanometres.
        n_medium: Real index of the surrounding medium.

    Returns:
        The efficiencies and asymmetry parameter, packaged with the size
        parameter and radius so downstream code never has to recompute them
        and risk disagreeing about units.
    """
    x = size_parameter(radius_m, wavelength_nm, n_medium=n_medium)
    m_rel = np.asarray(m_particle, dtype=complex) / n_medium
    q_ext, q_sca, g = solver.efficiencies(m_rel, x)
    x_b, q_ext, q_sca, g, r_b = np.broadcast_arrays(
        x,
        np.asarray(q_ext, dtype=float),
        np.asarray(q_sca, dtype=float),
        np.asarray(g, dtype=float),
        np.asarray(radius_m, dtype=float),
    )
    return MieProperties(x=x_b, q_ext=q_ext, q_sca=q_sca, g=g, radius_m=r_b)
