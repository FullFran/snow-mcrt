"""Where diffusion theory is right, and by how much it is wrong.

Every figure in ``docs/detectability.md`` rests on the diffusion
approximation: the sensitivity banana, the probing depth, the penetration
curves, the detection map. None of them had ever been checked against
transport. That is not a small omission -- diffusion is derived by assuming
the photon density is nearly isotropic, which is exactly the assumption that
fails where those figures are most interesting.

The 3-D engine is the reference. It samples the same optical properties with
no closure assumption at all, so the ratio ``R_MC(rho) / R_diffusion(rho)``
measures the approximation rather than any disagreement about the medium.

Two things had to line up before that ratio meant anything. Both engines must
see the same *surface* -- diffusion carries the index mismatch as an effective
internal reflection coefficient, and the Monte Carlo carries it as Fresnel at
each escape; the two agree to 0.07% and that agreement is asserted in
``tests/test_fresnel.py``. And both must see the same *source*: diffusion's
Green's function is a pencil beam, so the Monte Carlo runs collimated.

What comes out is not a pass or a fail. It is a range of validity, which is
the useful thing to publish.

**Measured on a Tesla P100 at two million photons**, across four snowpacks
spanning three orders of magnitude in co-albedo, the ratio has the same shape
every time and it is *not* monotonic:

===================  ==========================  ============================
band                 ``R_MC / R_diffusion``      reading
===================  ==========================  ============================
below 3 ``mfp'``     0.79 to 1.10                diffusion overestimates
3 to 12 ``mfp'``     0.96 to 1.11, median ~1.03  best agreement
12 to 50 ``mfp'``    0.91 to 0.97, median 0.93   diffusion overestimates ~7%
beyond 50 ``mfp'``   case-dependent              tail, and photon-starved
===================  ==========================  ============================

Two things follow, and the second is a trap.

Inside about three transport mean free paths diffusion is not inaccurate, it
is *inapplicable*: the photon has not yet forgotten which way it was going,
which is the entire premise. Quoting a worst case that includes that region
describes the premise failing rather than the approximation degrading, so
:meth:`DiffusionComparison.departure_between` exists to quote a band and
:meth:`worst_ratio_within` should never be reported on its own.

An empty bin is excluded by :meth:`DiffusionComparison.sampled`, but a bin
holding two photons is not empty and is not a measurement either. The far
tail runs out of photons long before it runs out of bins -- one clean-snow
bin came back at 0.085 -- so the tail is quoted as a range and never as a
number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.transport import TransportConfig
from snow_mcrt.domain.transport3d import log_radial_edges, run_transport_3d
from snow_mcrt.ports.backend import Backend


@dataclass(frozen=True)
class DiffusionComparison:
    """Radially resolved transport against radially resolved diffusion.

    Args:
        rho_m: Source-detector separation at each bin centre, metres.
        monte_carlo: ``R(rho)`` from the 3-D engine, m^-2.
        diffusion: ``R(rho)`` from the closed-form dipole solution, m^-2.
        transport_mfp_m: ``1 / (mu_a + mu_s')``, the length that sets where
            diffusion can be expected to apply at all.
        penetration_depth_m: ``1 / mu_eff``.
        reflected: Total fraction returned, from the engine.
        truncated: Fraction still in flight at the step budget. Anything
            appreciable here means the tail is unconverged, not that it is
            small.
        single_scattering_albedo: ``omega`` used.
        asymmetry: ``g`` used.
        extinction_coefficient: ``beta`` used, m^-1.
        surface_index: Refractive index of the medium.
        n_photons: Photons traced.
    """

    rho_m: np.ndarray
    monte_carlo: np.ndarray
    diffusion: np.ndarray
    transport_mfp_m: float
    penetration_depth_m: float
    reflected: float
    truncated: float
    single_scattering_albedo: float
    asymmetry: float
    extinction_coefficient: float
    surface_index: float
    n_photons: int

    @property
    def ratio(self) -> np.ndarray:
        """``R_MC / R_diffusion``. One where the approximation is exact."""
        return np.divide(
            self.monte_carlo,
            self.diffusion,
            out=np.full_like(self.monte_carlo, np.nan),
            where=self.diffusion > 0,
        )

    @property
    def rho_in_mfp(self) -> np.ndarray:
        """Separation in transport mean free paths -- the natural coordinate.

        Diffusion's validity is set by this, not by a distance in metres. A
        result quoted in centimetres does not transfer to another snowpack;
        one quoted in ``mfp'`` does.
        """
        return self.rho_m / self.transport_mfp_m

    def sampled(self) -> np.ndarray:
        """Bins the Monte Carlo actually populated.

        The far tail runs out of photons before it runs out of bins, and an
        empty bin is not a measurement of zero.
        """
        return (self.monte_carlo > 0) & np.isfinite(self.ratio)

    def worst_ratio_within(self, max_mfp: float) -> float:
        """Largest departure from one inside ``max_mfp`` transport lengths.

        Reported for completeness and misleading on its own: it is dominated
        by the bins nearest the source, where diffusion is not a poor
        approximation but an inapplicable one. Pair it with
        :meth:`departure_between`.
        """
        inside = self.sampled() & (self.rho_in_mfp <= max_mfp)
        if not inside.any():
            return float("nan")
        return float(np.max(np.abs(self.ratio[inside] - 1.0)))

    def departure_between(self, low_mfp: float, high_mfp: float) -> float:
        """Largest departure from one in a band, in transport mean free paths.

        The honest way to quote this comparison. ``departure_between(3, 12)``
        covers the range a source-detector measurement actually uses and
        excludes the near field, where the premise of diffusion theory has not
        been met rather than merely strained.
        """
        band = (
            self.sampled()
            & (self.rho_in_mfp >= low_mfp)
            & (self.rho_in_mfp <= high_mfp)
        )
        if not band.any():
            return float("nan")
        return float(np.max(np.abs(self.ratio[band] - 1.0)))

    def columns(self) -> dict[str, np.ndarray]:
        """Column name to values, in the order they are written to CSV."""
        return {
            "rho_m": self.rho_m,
            "rho_in_transport_mfp": self.rho_in_mfp,
            "reflectance_monte_carlo_per_m2": self.monte_carlo,
            "reflectance_diffusion_per_m2": self.diffusion,
            "ratio_mc_over_diffusion": self.ratio,
        }


def compare_with_diffusion(
    backend: Backend,
    single_scattering_albedo: float,
    asymmetry: float,
    extinction_coefficient: float,
    config: TransportConfig | None = None,
    surface_index: float = 1.31,
    inner_mfp: float = 1.0,
    outer_depths: float = 40.0,
    n_bins: int = 26,
) -> DiffusionComparison:
    """Run the 3-D engine and compare its profile against diffusion theory.

    Args:
        backend: Array backend.
        single_scattering_albedo: ``omega``.
        asymmetry: ``g``.
        extinction_coefficient: ``beta``, m^-1.
        config: Transport parameters.
        surface_index: Index of the medium; the same value is handed to the
            diffusion solution, so both see one surface.
        inner_mfp: Innermost bin edge, in transport mean free paths. Inside
            one of them diffusion is not merely inaccurate, it is undefined.
        outer_depths: Outermost bin edge, in penetration depths.
        n_bins: Radial bins.

    Returns:
        The two profiles and their ratio.
    """
    config = config or TransportConfig()
    parameters = DiffusionParameters.from_optical_properties(
        single_scattering_albedo,
        asymmetry,
        extinction_coefficient,
        refractive_index=surface_index,
    )
    mfp = parameters.transport_mean_free_path
    edges = log_radial_edges(
        inner_mfp * mfp, outer_depths * parameters.penetration_depth, n_bins
    )

    result = run_transport_3d(
        backend,
        extinction_coefficient,
        single_scattering_albedo,
        asymmetry,
        config=config,
        incidence="collimated",
        surface_index=surface_index,
        radial_edges_m=edges,
    )
    rho = result.bin_centres_m
    return DiffusionComparison(
        rho_m=rho,
        monte_carlo=result.reflectance,
        diffusion=np.asarray(parameters.diffuse_reflectance(rho), dtype=float),
        transport_mfp_m=mfp,
        penetration_depth_m=parameters.penetration_depth,
        reflected=result.reflected,
        truncated=result.truncated,
        single_scattering_albedo=single_scattering_albedo,
        asymmetry=asymmetry,
        extinction_coefficient=extinction_coefficient,
        surface_index=surface_index,
        n_photons=config.n_photons,
    )
