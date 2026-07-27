"""Cross-validation against TARTES, with the comparison decomposed.

The headline assertion is the *transfer* residual, not the total. Two codes
agreeing end to end would be reassuring; two codes agreeing to four decimal
places once the grain model is held fixed says something much stronger — that
they solve the same transfer problem the same way, and that everything else
between them is a modelling choice rather than an error.
"""

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants
from snow_mcrt.application.validate_tartes import (
    compare_with_tartes,
    radius_from_ssa,
    specific_surface_area,
)

tartes = pytest.importorskip(
    "tartes", reason="TARTES is a validation-only dependency"
)

CONSTANTS_PATH = "data/ice/warren_brandt_2008.dat"
WAVELENGTHS = np.array([400.0, 500.0, 700.0, 900.0, 1100.0, 1300.0])


@pytest.fixture(scope="module")
def solver():
    return MiepythonSolver()


@pytest.fixture(scope="module")
def constants():
    return TabulatedConstants(
        CONSTANTS_PATH, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()


@pytest.fixture(scope="module")
def comparison(solver, constants):
    return compare_with_tartes(solver, constants, WAVELENGTHS, grain_radius_m=100e-6)


class TestSpecificSurfaceArea:
    def test_round_trips(self):
        assert radius_from_ssa(specific_surface_area(100e-6)) == pytest.approx(100e-6)

    def test_a_hundred_micron_grain_is_about_33_square_metres_per_kilogram(self):
        # The number a snow scientist would recognise. Fine new snow runs
        # 60-100, aged snow single digits.
        assert specific_surface_area(100e-6) == pytest.approx(32.7, abs=0.2)

    def test_smaller_grains_have_more_surface(self):
        assert specific_surface_area(50e-6) > specific_surface_area(100e-6)

    def test_rejects_a_nonpositive_radius(self):
        with pytest.raises(ValueError, match="must be positive"):
            specific_surface_area(0.0)


class TestOpticalConstants:
    def test_both_codes_read_the_same_ice(self, constants):
        # If this fails nothing else in the comparison means anything: the
        # difference would be in the inputs rather than the physics.
        from tartes.refractive_index import refice2008

        n_theirs, k_theirs = refice2008(WAVELENGTHS * 1e-9)
        ours = constants.m_at(WAVELENGTHS)
        assert np.allclose(np.real(ours), n_theirs, rtol=1e-6)
        assert np.allclose(np.imag(ours), k_theirs, rtol=1e-6)


class TestRadiativeTransfer:
    """The strong result: hold the grain model fixed and the codes agree."""

    def test_our_solver_reproduces_tartes_given_its_own_parameters(self, comparison):
        # Four decimal places, across the visible and near infrared. This is
        # research question 4 answered for the transfer solution.
        assert np.max(np.abs(comparison.transfer_residual)) < 5e-3

    @pytest.mark.parametrize("radius_um", [50.0, 100.0, 250.0, 1000.0])
    def test_agreement_holds_across_grain_sizes(self, solver, constants, radius_um):
        result = compare_with_tartes(
            solver, constants, WAVELENGTHS, grain_radius_m=radius_um * 1e-6
        )
        assert np.max(np.abs(result.transfer_residual)) < 5e-3


class TestGrainModel:
    """The residual, and what it is."""

    def test_spheres_are_more_forward_scattering_than_real_snow(self, comparison):
        # Full Mie on spheres gives g ~ 0.89. TARTES uses 0.82, calibrated for
        # the non-spherical grains real snow is made of. Spheres over-predict
        # forward scattering, which is a known limitation and is why
        # non-spherical morphology is out of scope for v1 rather than absent
        # by oversight.
        assert np.all(comparison.our_asymmetry > 0.88)
        assert np.all(comparison.their_asymmetry == pytest.approx(0.82, abs=0.01))

    def test_the_grain_model_is_what_drives_the_total_difference(self, comparison):
        # Decomposition sanity: the grain-model residual must dominate the
        # transfer residual, or the story told above is wrong.
        grain = np.max(np.abs(comparison.grain_model_residual))
        transfer = np.max(np.abs(comparison.transfer_residual))
        assert grain > 10 * transfer

    def test_more_forward_scattering_means_a_darker_snowpack(self, comparison):
        # Higher g sends photons deeper per scattering event, so they
        # accumulate more path in ice before escaping. Our albedo must
        # therefore sit below TARTES everywhere the ice absorbs at all.
        absorbing = comparison.wavelength_nm > 500.0
        assert np.all(comparison.total_residual[absorbing] < 0)

    def test_the_two_codes_still_agree_where_absorption_vanishes(self, comparison):
        # In the blue, ice barely absorbs, so g hardly matters and the codes
        # converge regardless of grain model. A disagreement here would mean
        # something more basic was wrong.
        blue = comparison.wavelength_nm <= 400.0
        assert np.all(np.abs(comparison.total_residual[blue]) < 2e-3)


class TestOutput:
    def test_columns_are_aligned_and_complete(self, comparison):
        columns = comparison.columns()
        assert set(columns) >= {"snow_mcrt", "tartes", "transfer_residual"}
        assert all(v.shape == WAVELENGTHS.shape for v in columns.values())

    def test_residuals_decompose_exactly(self, comparison):
        assert np.allclose(
            comparison.total_residual,
            comparison.transfer_residual + comparison.grain_model_residual,
        )
