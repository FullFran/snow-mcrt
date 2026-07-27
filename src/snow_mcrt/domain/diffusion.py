"""Diffusion theory for a semi-infinite snowpack.

Where the transport module traces photons, this one describes what the photon
cloud does once it has forgotten where it came from. After a few transport mean
free paths — under two millimetres in snow — the radiance field is diffusive,
and diffusion theory then gives closed forms for the quantities a
surface-based instrument would actually measure:

- the fluence at depth, from a source on the surface
- the diffuse reflectance as a function of source-detector separation
- the **sensitivity kernel**: where between a source and a detector the
  detected light has actually been

That last one is what makes buried-object detection a well-posed question
rather than a hopeful one. A source and detector separated by ``rho`` do not
sample the whole snowpack; they sample a banana-shaped volume between them,
and an object outside it is invisible no matter how strongly it absorbs.

These are also the oracle for the eventual three-dimensional Monte Carlo, in
exactly the way :mod:`snow_mcrt.domain.analytic` is the oracle for the
one-dimensional one. Diffusion theory is an approximation, but snow is a
near-ideal diffuser -- ``omega`` within a millionth of unity, ``g`` around
0.89 -- so it holds better here than in the tissue optics it was developed for.

References:
    Patterson, Chance & Wilson (1989), Appl. Opt. 28, 2331 -- time-resolved
    reflectance and the extrapolated boundary.
    Haskell et al. (1994), J. Opt. Soc. Am. A 11, 2727 -- boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiffusionParameters:
    """Diffusion description of a homogeneous semi-infinite medium.

    Args:
        absorption_coefficient: ``mu_a``, m^-1.
        reduced_scattering_coefficient: ``mu_s' = mu_s (1 - g)``, m^-1.
        refractive_index: Real index of the medium, used for the internal
            reflection that sets the extrapolated boundary.
    """

    absorption_coefficient: float
    reduced_scattering_coefficient: float
    refractive_index: float = 1.31

    def __post_init__(self) -> None:
        if self.absorption_coefficient < 0:
            raise ValueError("absorption coefficient must be non-negative")
        if self.reduced_scattering_coefficient <= 0:
            raise ValueError("reduced scattering coefficient must be positive")
        if self.refractive_index < 1:
            raise ValueError("refractive index must be at least 1")

    @classmethod
    def from_optical_properties(
        cls,
        single_scattering_albedo: float,
        asymmetry: float,
        extinction_coefficient: float,
        refractive_index: float = 1.31,
    ) -> DiffusionParameters:
        """Build from the quantities the rest of the engine speaks."""
        omega = float(single_scattering_albedo)
        g = float(asymmetry)
        beta = float(extinction_coefficient)
        return cls(
            absorption_coefficient=beta * (1.0 - omega),
            reduced_scattering_coefficient=beta * omega * (1.0 - g),
            refractive_index=refractive_index,
        )

    @property
    def transport_mean_free_path(self) -> float:
        """``1 / mu_s'``, the distance over which direction is forgotten."""
        return 1.0 / self.reduced_scattering_coefficient

    @property
    def diffusion_coefficient(self) -> float:
        """``D = 1 / (3 (mu_a + mu_s'))``, in metres."""
        return 1.0 / (
            3.0
            * (self.absorption_coefficient + self.reduced_scattering_coefficient)
        )

    @property
    def effective_attenuation(self) -> float:
        """``mu_eff = sqrt(mu_a / D)``, m^-1. Its reciprocal is the e-folding depth."""
        return float(np.sqrt(self.absorption_coefficient / self.diffusion_coefficient))

    @property
    def penetration_depth(self) -> float:
        """``1 / mu_eff``, metres. Infinite for a non-absorbing medium."""
        if self.absorption_coefficient == 0:
            return np.inf
        return 1.0 / self.effective_attenuation

    @property
    def internal_reflection(self) -> float:
        """Effective internal reflection coefficient at the surface.

        Empirical fit widely used in diffuse optics. Snow at ``n = 1.31``
        reflects about half the diffuse flux arriving at the boundary back
        into the medium, which pushes the extrapolated boundary well outside
        the surface and is not a detail that can be dropped.
        """
        n = self.refractive_index
        return -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n

    @property
    def extrapolated_boundary(self) -> float:
        """``z_b``: how far above the surface the fluence extrapolates to zero."""
        reflection = self.internal_reflection
        return (
            2.0
            * self.diffusion_coefficient
            * (1.0 + reflection)
            / (1.0 - reflection)
        )

    def fluence(self, rho: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Fluence at radial distance ``rho`` and depth ``z`` from a surface source.

        The isotropic source sits one transport mean free path down, with a
        negative image above the extrapolated boundary enforcing the boundary
        condition.

        Args:
            rho: Radial distance from the source, metres.
            z: Depth below the surface, metres.

        Returns:
            Fluence in arbitrary units, positive inside the medium.
        """
        rho = np.asarray(rho, dtype=float)
        z = np.asarray(z, dtype=float)
        z0 = self.transport_mean_free_path
        zb = self.extrapolated_boundary
        mu_eff = self.effective_attenuation
        D = self.diffusion_coefficient

        r_real = np.sqrt(rho**2 + (z - z0) ** 2)
        r_image = np.sqrt(rho**2 + (z + z0 + 2.0 * zb) ** 2)
        # Guard the singularity at the source itself; diffusion theory does not
        # apply within a transport mean free path anyway.
        r_real = np.maximum(r_real, 1e-9)
        r_image = np.maximum(r_image, 1e-9)
        return (
            np.exp(-mu_eff * r_real) / r_real
            - np.exp(-mu_eff * r_image) / r_image
        ) / (4.0 * np.pi * D)

    def diffuse_reflectance(self, rho: np.ndarray | float) -> np.ndarray:
        """Radially resolved diffuse reflectance at the surface.

        Args:
            rho: Source-detector separation, metres.

        Returns:
            Reflectance per unit area, arbitrary units.
        """
        rho = np.asarray(rho, dtype=float)
        z0 = self.transport_mean_free_path
        zb = self.extrapolated_boundary
        mu_eff = self.effective_attenuation

        r_real = np.sqrt(rho**2 + z0**2)
        r_image = np.sqrt(rho**2 + (z0 + 2.0 * zb) ** 2)
        return (
            z0 * (mu_eff + 1.0 / r_real) * np.exp(-mu_eff * r_real) / r_real**2
            + (z0 + 2.0 * zb)
            * (mu_eff + 1.0 / r_image)
            * np.exp(-mu_eff * r_image)
            / r_image**2
        ) / (4.0 * np.pi)

    def sensitivity_kernel(
        self,
        separation: float,
        x: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Where the detected light has been — the "banana".

        The product of the fluence from the source and the fluence that would
        reach the detector, evaluated throughout the medium. An absorbing
        object perturbs the measurement in proportion to this, so it maps
        exactly what a given geometry can and cannot see.

        Args:
            separation: Source-detector separation ``rho``, metres. The source
                sits at ``x = 0``, the detector at ``x = separation``.
            x: Horizontal coordinates, metres.
            z: Depths, metres.

        Returns:
            Kernel on the ``(z, x)`` grid, normalised to a maximum of 1.
        """
        if separation <= 0:
            raise ValueError("source-detector separation must be positive")
        x_grid, z_grid = np.meshgrid(np.asarray(x), np.asarray(z))
        from_source = self.fluence(np.abs(x_grid), z_grid)
        to_detector = self.fluence(np.abs(x_grid - separation), z_grid)
        kernel = from_source * to_detector
        peak = kernel.max()
        return kernel / peak if peak > 0 else kernel

    def probing_depth(self, separation: float, max_depth: float | None = None) -> float:
        """Depth of peak sensitivity midway between source and detector.

        The number that decides instrument geometry. Tissue optics quotes
        ``rho/2`` as a rule of thumb; snow gives closer to ``rho/3``.

        Args:
            separation: Source-detector separation, metres.
            max_depth: Search limit. Defaults to twice the separation.

        Returns:
            Depth of maximum sensitivity, metres.
        """
        if separation <= 0:
            raise ValueError("source-detector separation must be positive")
        limit = max_depth if max_depth is not None else 2.0 * separation
        depths = np.linspace(1e-4, limit, 4000)
        midline = self.fluence(separation / 2.0, depths) ** 2
        return float(depths[int(np.argmax(midline))])


def two_way_detection_depth(
    penetration_depth: float, signal_fraction: float = 0.01
) -> float:
    """Depth at which a two-way signal falls to ``signal_fraction``.

    Light must reach the object and come back, so the round trip attenuates as
    ``exp(-2 z / d)``. Solving for the stated fraction:

    .. math:: z = \\frac{d}{2} \\ln\\frac{1}{f}

    A coarse figure of merit rather than a detection threshold — it ignores
    the object's size and contrast entirely — but it is the right scale for
    asking whether a burial depth is plausible at all.
    """
    if penetration_depth <= 0:
        raise ValueError("penetration depth must be positive")
    if not 0 < signal_fraction < 1:
        raise ValueError("signal fraction must lie in (0, 1)")
    return 0.5 * penetration_depth * np.log(1.0 / signal_fraction)
