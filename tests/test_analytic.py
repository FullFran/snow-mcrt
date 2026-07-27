"""The analytic oracle: limits it must satisfy, and values a snow reader knows."""

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.domain.analytic import (
    asymptotic_extinction_coefficient,
    delta_eddington_scaling,
    e_folding_depth,
    semi_infinite_albedo,
    similarity_parameter,
)
from snow_mcrt.domain.mie import compute_mie_properties

ICE_M_VISIBLE = 1.3105 + 2.0e-9j
ICE_M_NEAR_IR = 1.2985 + 1.3e-5j
SEASONAL_SNOW_DENSITY = 300.0


@pytest.fixture
def solver():
    return MiepythonSolver()


def snowpack(solver, m, radius_m, wavelength_nm):
    """Single-scattering albedo, asymmetry and extinction for a snowpack."""
    props = compute_mie_properties(solver, m, radius_m, wavelength_nm)
    return (
        props.single_scattering_albedo[0],
        props.g[0],
        props.extinction_coefficient_from_density(SEASONAL_SNOW_DENSITY)[0],
    )


class TestLimits:
    def test_a_conservative_medium_reflects_everything(self):
        # No absorption means no sink. Whatever goes in must come back out,
        # however many times it scatters on the way.
        assert semi_infinite_albedo(1.0, 0.89) == pytest.approx(1.0)

    def test_a_purely_absorbing_medium_reflects_nothing(self):
        assert semi_infinite_albedo(0.0, 0.0) == pytest.approx(0.0)

    def test_albedo_increases_with_single_scattering_albedo(self):
        omega = np.array([0.1, 0.5, 0.9, 0.99, 0.9999])
        alpha = semi_infinite_albedo(omega, 0.89)
        assert np.all(np.diff(alpha) > 0)

    def test_albedo_stays_within_the_unit_interval(self):
        omega = np.linspace(0.0, 1.0, 101)
        alpha = semi_infinite_albedo(omega, 0.89)
        assert np.all((alpha >= 0.0) & (alpha <= 1.0))

    def test_forward_scattering_lowers_the_albedo(self):
        # Forward scattering carries photons deeper per event, so each one
        # accumulates more path in ice before it can escape.
        assert semi_infinite_albedo(0.99, 0.9) < semi_infinite_albedo(0.99, 0.1)


class TestSimilarity:
    def test_vanishes_for_a_conservative_medium(self):
        assert similarity_parameter(1.0, 0.89) == pytest.approx(0.0)

    def test_two_snowpacks_with_equal_s_share_an_albedo(self):
        # Similarity theory is what lets grain shape be folded into an
        # effective radius, and what lets v1 get away with spheres.
        s_target = similarity_parameter(0.99, 0.89)
        for g in (0.5, 0.7, 0.89):
            omega = (1.0 - s_target**2) / (1.0 - s_target**2 * g)
            assert similarity_parameter(omega, g) == pytest.approx(s_target)
            assert semi_infinite_albedo(
                omega, g, delta_scaled=False
            ) == pytest.approx(
                semi_infinite_albedo(0.99, 0.89, delta_scaled=False)
            )


class TestDeltaEddingtonScaling:
    def test_leaves_isotropic_scattering_untouched(self):
        omega, g = delta_eddington_scaling(0.9, 0.0)
        assert omega == pytest.approx(0.9)
        assert g == pytest.approx(0.0)

    def test_reduces_both_asymmetry_and_albedo(self):
        omega, g = delta_eddington_scaling(0.99, 0.89)
        assert g < 0.89
        assert omega < 0.99

    def test_preserves_a_conservative_medium(self):
        # Truncating the forward peak must not manufacture absorption.
        omega, _ = delta_eddington_scaling(1.0, 0.89)
        assert omega == pytest.approx(1.0)


class TestSnowValues:
    """Numbers a snow-optics reader should recognise on sight."""

    def test_clean_fine_snow_is_near_white_in_the_visible(self, solver):
        omega, g, _ = snowpack(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        alpha = float(semi_infinite_albedo(omega, g))
        assert 0.97 < alpha < 0.999

    def test_near_infrared_albedo_is_far_lower(self, solver):
        # At 1300 nm ice absorbs some four orders of magnitude more strongly
        # than at 500 nm, and snow that looks white to the eye is dark here.
        omega, g, _ = snowpack(solver, ICE_M_NEAR_IR, 100e-6, 1300.0)
        alpha = float(semi_infinite_albedo(omega, g))
        assert 0.3 < alpha < 0.75

    def test_coarser_grains_darken_the_near_infrared(self, solver):
        # The grain-size signal remote sensing actually exploits.
        fine = semi_infinite_albedo(*snowpack(solver, ICE_M_NEAR_IR, 50e-6, 1300.0)[:2])
        coarse = semi_infinite_albedo(
            *snowpack(solver, ICE_M_NEAR_IR, 1000e-6, 1300.0)[:2]
        )
        assert coarse < fine

    def test_grain_size_matters_far_less_in_the_visible(self, solver):
        # The counterpart, and the reason the two bands answer different
        # questions: near-infrared albedo is a grain-size diagnostic, visible
        # albedo is an impurity one. Stated as a ratio of spreads rather than
        # an absolute threshold, because the absolute numbers move with the
        # ice constants while the ordering is the physics.
        def spread(m, wavelength_nm):
            fine = semi_infinite_albedo(*snowpack(solver, m, 50e-6, wavelength_nm)[:2])
            coarse = semi_infinite_albedo(
                *snowpack(solver, m, 1000e-6, wavelength_nm)[:2]
            )
            return float(fine) - float(coarse)

        visible = spread(ICE_M_VISIBLE, 500.0)
        near_ir = spread(ICE_M_NEAR_IR, 1300.0)
        assert visible > 0
        assert near_ir > 10 * visible


class TestEFoldingDepth:
    def test_visible_light_penetrates_of_order_ten_centimetres(self, solver):
        # Libois et al. (2013) measure e-folding depths of this order in the
        # visible. Research question 3 compares against them directly.
        omega, g, beta = snowpack(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        depth_cm = float(e_folding_depth(omega, g, beta)) * 100.0
        assert 2.0 < depth_cm < 50.0

    def test_near_infrared_light_barely_penetrates(self, solver):
        omega, g, beta = snowpack(solver, ICE_M_NEAR_IR, 100e-6, 1300.0)
        depth_mm = float(e_folding_depth(omega, g, beta)) * 1000.0
        assert 0.5 < depth_mm < 20.0

    def test_light_penetrates_far_deeper_than_one_mean_free_path(self, solver):
        # Photons scatter thousands of times before being absorbed, so the
        # diffuse flux decays orders of magnitude more slowly than beta.
        omega, g, beta = snowpack(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        assert float(e_folding_depth(omega, g, beta)) > 100.0 / beta

    def test_a_conservative_medium_never_extinguishes(self):
        assert np.isinf(e_folding_depth(1.0, 0.89, 5000.0))

    def test_extinction_rate_scales_with_the_extinction_coefficient(self):
        single = asymptotic_extinction_coefficient(0.99, 0.5, 1000.0)
        double = asymptotic_extinction_coefficient(0.99, 0.5, 2000.0)
        assert double == pytest.approx(2 * single)


class TestValidation:
    def test_rejects_an_out_of_range_albedo(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            semi_infinite_albedo(1.5, 0.5)

    def test_rejects_an_out_of_range_asymmetry_parameter(self):
        with pytest.raises(ValueError, match=r"\(-1, 1\)"):
            delta_eddington_scaling(0.9, 1.0)

    def test_rejects_a_negative_extinction_coefficient(self):
        with pytest.raises(ValueError, match="non-negative"):
            asymptotic_extinction_coefficient(0.9, 0.5, -1.0)
