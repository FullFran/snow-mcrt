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


@dataclass(frozen=True)
class LogNormalGrainSizes:
    """A log-normal distribution of grain radii.

    Monodisperse spheres are not merely an idealisation here, they are an
    actively misleading one. A perfect sphere supports morphology-dependent
    resonances -- whispering-gallery modes -- that spike absorption by an order
    of magnitude at isolated size parameters. Measured on the Warren & Brandt
    constants at 676.7 nm with 100 um grains: the co-albedo jumps 13.4x above
    its background over a resonance only ``0.48`` wide in ``x``.

    Those spikes are real physics for one perfect sphere and pure fiction for a
    snowpack. Real grains vary in size, and the averaging destroys the
    resonances completely -- a geometric standard deviation of 1.05, a 5%
    spread, already restores the smooth background. Real snow is nearer 1.5.

    Args:
        median_radius_m: Median (geometric mean) radius in metres.
        sigma_g: Geometric standard deviation. Must be >= 1; exactly 1 is
            monodisperse and reproduces the single-radius result, resonances
            and all.
        n_quadrature: Points used to integrate over the distribution.
    """

    median_radius_m: float
    sigma_g: float = 1.5
    n_quadrature: int = 16
    n_sigma: float = 3.0

    def __post_init__(self) -> None:
        if self.median_radius_m <= 0:
            raise ValueError("median radius must be positive")
        if self.sigma_g < 1.0:
            raise ValueError(
                "geometric standard deviation must be >= 1; "
                "1 means monodisperse"
            )
        if self.n_quadrature < 2:
            raise ValueError("need at least two quadrature points")
        if self.n_quadrature % 2 != 0:
            # Not a stylistic preference. An odd, symmetric node count puts a
            # node exactly on the median radius and gives it the largest
            # weight in the distribution. When that radius happens to sit on a
            # resonance -- as 100 um grains do at 676.7 nm -- the resonance
            # survives the averaging it was supposed to be destroyed by.
            #
            # Measured at that wavelength, against smooth neighbours of
            # 4.4e-5 and 5.5e-5: n=17 gives 6.6e-5 and n=33 gives 5.7e-5,
            # while every even count from 16 to 64 lands on 4.7e-5.
            raise ValueError(
                f"n_quadrature must be even, got {self.n_quadrature}. An odd "
                f"node count samples the median radius with maximum weight, "
                f"which reintroduces the sphere resonances this distribution "
                f"exists to average away."
            )

    def radii_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature radii and their number-fraction weights.

        Deterministic, so a run is reproducible from its manifest. Monte Carlo
        sampling of the size distribution would add a second, unnecessary
        source of noise on top of the photon transport.
        """
        if self.sigma_g == 1.0:
            return np.array([self.median_radius_m]), np.array([1.0])
        ln_sigma = np.log(self.sigma_g)
        # Three geometric standard deviations, not four. The tail beyond it
        # carries a thousandth of the number, but its cost is brutal: at
        # sigma_g = 1.5 the fourth deviation reaches five times the median
        # radius, which for millimetre grains in the near ultraviolet means a
        # size parameter above 1e5. The Mie series there costs more than
        # everything else in the run combined and contributes nothing that
        # survives rounding.
        ln_r = np.linspace(
            np.log(self.median_radius_m) - self.n_sigma * ln_sigma,
            np.log(self.median_radius_m) + self.n_sigma * ln_sigma,
            self.n_quadrature,
        )
        weights = np.exp(
            -0.5 * ((ln_r - np.log(self.median_radius_m)) / ln_sigma) ** 2
        )
        return np.exp(ln_r), weights / weights.sum()

    @property
    def mean_cube_radius(self) -> float:
        """``<r^3>``, which is what fixes number density from bulk density."""
        radii, weights = self.radii_and_weights()
        return float(np.sum(weights * radii**3))


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
