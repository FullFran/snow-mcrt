"""Cross-validation against TARTES — an independent published implementation.

TARTES (Libois, Picard et al., 2013) solves the same problem by a different
route, and it is already in this project's bibliography. Comparing against it
is research question 4.

**The comparison is decomposed, and that is the whole point.** Two codes
disagreeing on an albedo curve tells you almost nothing: the difference could
live in the ice constants, the single-scattering model, or the radiative
transfer solution, and a single number cannot say which. So this module runs
three comparisons instead of one:

1. **Optical constants.** Both read Warren & Brandt (2008). If these differ,
   nothing downstream means anything.
2. **Radiative transfer.** Feed TARTES's own single-scattering parameters into
   *our* solver. Any difference here is a difference in how the two codes
   solve the transfer problem, with the grain model held fixed.
3. **Whole pipeline.** Each code end to end from its own grain model.

The residual between (2) and (3) is exactly the grain model, and for snow that
means the sphere assumption: full Mie on spheres gives ``g ~ 0.89``, while
TARTES uses ``g0 = 0.82``, calibrated for the non-spherical grains real snow
is made of. Non-spherical morphology is explicitly out of scope for v1, so
this residual is a measurement of a known limitation rather than a defect.

TARTES parameterises grain size by specific surface area rather than radius.
For spheres ``SSA = 3 / (rho_ice * r)``, which is the conversion used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snow_mcrt.domain.analytic import similarity_scaled_albedo
from snow_mcrt.domain.medium import SnowLayer, compute_layer_properties
from snow_mcrt.domain.mie import ICE_DENSITY
from snow_mcrt.domain.optics import OpticalConstants
from snow_mcrt.ports.mie_solver import MieSolver


def specific_surface_area(radius_m: float) -> float:
    """Convert sphere radius to specific surface area, m^2/kg.

    .. math:: \\mathrm{SSA} = \\frac{3}{\\rho_{ice} r}

    The quantity snow science actually measures, because it survives the fact
    that real grains are not spheres.
    """
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    return 3.0 / (ICE_DENSITY * radius_m)


def radius_from_ssa(ssa: float) -> float:
    """Inverse of :func:`specific_surface_area`, the optically equivalent radius."""
    if ssa <= 0:
        raise ValueError("specific surface area must be positive")
    return 3.0 / (ICE_DENSITY * ssa)


@dataclass(frozen=True)
class TartesComparison:
    """One decomposed comparison over a wavelength grid.

    Args:
        wavelength_nm: Wavelength grid.
        grain_radius_m: Sphere radius used on our side.
        ours: Our albedo, our grain model, end to end.
        theirs: TARTES albedo, its grain model, end to end.
        ours_with_their_single_scattering: Our transfer solver fed TARTES's
            single-scattering parameters. Isolates the transfer solution.
        our_asymmetry: Asymmetry parameter from full Mie on spheres.
        their_asymmetry: The value TARTES uses.
        our_co_albedo: ``1 - omega`` from full Mie on spheres.
        their_co_albedo: The same from TARTES.
    """

    wavelength_nm: np.ndarray
    grain_radius_m: float
    ours: np.ndarray
    theirs: np.ndarray
    ours_with_their_single_scattering: np.ndarray
    our_asymmetry: np.ndarray
    their_asymmetry: np.ndarray
    our_co_albedo: np.ndarray
    their_co_albedo: np.ndarray

    @property
    def transfer_residual(self) -> np.ndarray:
        """Difference attributable to the radiative transfer solution alone."""
        return self.ours_with_their_single_scattering - self.theirs

    @property
    def grain_model_residual(self) -> np.ndarray:
        """Difference attributable to the grain model alone."""
        return self.ours - self.ours_with_their_single_scattering

    @property
    def total_residual(self) -> np.ndarray:
        """End-to-end difference between the two codes."""
        return self.ours - self.theirs

    def columns(self) -> dict[str, np.ndarray]:
        """Column name to values, for CSV output."""
        return {
            "wavelength_nm": self.wavelength_nm,
            "snow_mcrt": self.ours,
            "tartes": self.theirs,
            "snow_mcrt_rt_with_tartes_ssp": self.ours_with_their_single_scattering,
            "transfer_residual": self.transfer_residual,
            "grain_model_residual": self.grain_model_residual,
            "total_residual": self.total_residual,
            "our_asymmetry": self.our_asymmetry,
            "their_asymmetry": self.their_asymmetry,
            "our_co_albedo": self.our_co_albedo,
            "their_co_albedo": self.their_co_albedo,
        }


def compare_with_tartes(
    solver: MieSolver,
    constants: OpticalConstants,
    wavelength_nm: np.ndarray,
    grain_radius_m: float = 100e-6,
    snow_density: float = 300.0,
) -> TartesComparison:
    """Run the three-way decomposed comparison.

    Args:
        solver: Mie solver.
        constants: Ice optical constants, ours.
        wavelength_nm: Wavelength grid in nanometres.
        grain_radius_m: Sphere radius.
        snow_density: Bulk density, kg/m^3.

    Returns:
        The comparison, with the transfer and grain-model residuals separated.

    Raises:
        ImportError: If TARTES is not installed. It is a validation-only
            dependency, so the engine never requires it.
    """
    try:
        import tartes
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "TARTES is required for this comparison: pip install tartes. It is "
            "a validation dependency only; the engine does not need it."
        ) from exc

    wavelengths = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    wavelengths_m = wavelengths * 1e-9
    ssa = specific_surface_area(grain_radius_m)

    layer = SnowLayer(grain_radius_m, snow_density)
    props = compute_layer_properties(
        solver, layer, constants.m_at(wavelengths), wavelengths
    )
    ours = np.atleast_1d(
        similarity_scaled_albedo(props.single_scattering_albedo, props.asymmetry)
    )

    # "w2008" selects the same Warren & Brandt compilation this project uses,
    # so the constants are not a free variable in the comparison.
    their_omega, their_g = tartes.single_scattering_optical_parameters(
        wavelengths_m, "w2008", ssa
    )
    their_omega = np.atleast_1d(their_omega)
    their_g = np.atleast_1d(their_g)

    theirs = np.atleast_1d(
        tartes.albedo(
            wavelengths_m, SSA=ssa, density=snow_density, refrac_index="w2008"
        )
    )
    bridged = np.atleast_1d(similarity_scaled_albedo(their_omega, their_g))

    return TartesComparison(
        wavelength_nm=wavelengths,
        grain_radius_m=grain_radius_m,
        ours=ours,
        theirs=theirs,
        ours_with_their_single_scattering=bridged,
        our_asymmetry=props.asymmetry,
        their_asymmetry=their_g,
        our_co_albedo=1.0 - props.single_scattering_albedo,
        their_co_albedo=1.0 - their_omega,
    )
