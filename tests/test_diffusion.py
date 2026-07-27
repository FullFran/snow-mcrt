"""Diffusion theory: limits it must satisfy, and the geometry it predicts."""

import numpy as np
import pytest

from snow_mcrt.domain.analytic import e_folding_depth
from snow_mcrt.domain.diffusion import (
    DiffusionParameters,
    two_way_detection_depth,
)

# Clean-ish alpine snow, 100 um grains, 300 kg/m^3, in the blue.
SNOW = DiffusionParameters(
    absorption_coefficient=2.6e-3, reduced_scattering_coefficient=548.0
)


class TestDerivedQuantities:
    def test_transport_mean_free_path_is_millimetric(self):
        # 1/548 m. Snow forgets a photon's direction in under two millimetres,
        # which is why anything deeper than a centimetre is purely diffusive.
        assert 1e-3 < SNOW.transport_mean_free_path < 3e-3

    def test_penetration_depth_is_decimetric_in_the_blue(self):
        assert 0.1 < SNOW.penetration_depth < 1.0

    def test_a_non_absorbing_medium_never_attenuates(self):
        clear = DiffusionParameters(0.0, 548.0)
        assert np.isinf(clear.penetration_depth)

    def test_more_absorption_means_shallower_penetration(self):
        shallow = DiffusionParameters(1e-2, 548.0)
        assert shallow.penetration_depth < SNOW.penetration_depth

    def test_penetration_scales_as_the_inverse_square_root_of_absorption(self):
        # mu_eff = sqrt(mu_a / D), so quadrupling absorption halves the depth.
        quadrupled = DiffusionParameters(4 * 2.6e-3, 548.0)
        assert quadrupled.penetration_depth == pytest.approx(
            SNOW.penetration_depth / 2, rel=0.02
        )


class TestAgreementWithTheTransportModule:
    def test_penetration_depth_matches_the_asymptotic_formula(self):
        # domain.analytic computes the same quantity from (omega, g, beta) by
        # a different route. They must agree, or one of them is wrong.
        omega, g, beta = 1.0 - 5.31e-7, 0.889, 4950.0
        params = DiffusionParameters.from_optical_properties(omega, g, beta)
        from_analytic = float(e_folding_depth(omega, g, beta, delta_scaled=False))
        assert params.penetration_depth == pytest.approx(from_analytic, rel=0.02)

    def test_optical_properties_round_trip(self):
        omega, g, beta = 0.999, 0.85, 5000.0
        params = DiffusionParameters.from_optical_properties(omega, g, beta)
        assert params.absorption_coefficient == pytest.approx(beta * (1 - omega))
        assert params.reduced_scattering_coefficient == pytest.approx(
            beta * omega * (1 - g)
        )


class TestBoundary:
    def test_snow_reflects_much_of_the_diffuse_flux_internally(self):
        # n = 1.31 gives an internal reflection coefficient near 0.5. Treating
        # the surface as non-reflecting would move the extrapolated boundary by
        # a factor of three and change every near-field prediction.
        assert 0.4 < SNOW.internal_reflection < 0.6

    def test_the_extrapolated_boundary_sits_above_the_surface(self):
        assert SNOW.extrapolated_boundary > 0

    def test_it_is_a_few_transport_mean_free_paths(self):
        ratio = SNOW.extrapolated_boundary / SNOW.transport_mean_free_path
        assert 0.5 < ratio < 5.0


class TestFluence:
    def test_falls_off_with_depth(self):
        depths = np.array([0.01, 0.05, 0.1, 0.3])
        values = SNOW.fluence(0.0, depths)
        assert np.all(np.diff(values) < 0)

    def test_falls_off_with_radial_distance(self):
        values = SNOW.fluence(np.array([0.05, 0.2, 0.5]), 0.05)
        assert np.all(np.diff(values) < 0)

    def test_is_positive_inside_the_medium(self):
        rho = np.linspace(0.0, 1.0, 30)
        z = np.linspace(0.005, 0.5, 30)
        grid = SNOW.fluence(rho[None, :], z[:, None])
        assert np.all(grid > 0)


class TestDiffuseReflectance:
    def test_falls_steeply_with_separation(self):
        rho = np.array([0.1, 0.4, 1.0, 1.5])
        values = SNOW.diffuse_reflectance(rho)
        assert np.all(np.diff(values) < 0)
        # Five orders of magnitude from 10 cm to 150 cm. Steep, but from a
        # very large number -- the photon budget is not the constraint.
        assert values[0] / values[-1] > 1e4

    def test_is_positive_everywhere(self):
        assert np.all(SNOW.diffuse_reflectance(np.linspace(0.01, 2.0, 50)) > 0)


class TestSensitivityKernel:
    """The banana, and the geometry it forces on any instrument."""

    @pytest.mark.parametrize("separation", [0.1, 0.2, 0.4, 0.6])
    def test_probing_depth_is_about_a_third_of_the_separation(self, separation):
        # Tissue optics quotes rho/2. Snow's parameters give closer to rho/3,
        # and the difference decides how far apart an instrument's source and
        # detector have to sit.
        ratio = SNOW.probing_depth(separation) / separation
        assert 0.25 < ratio < 0.40

    def test_deeper_probing_requires_wider_separation(self):
        assert SNOW.probing_depth(0.6) > SNOW.probing_depth(0.2)

    def test_the_kernel_is_symmetric_about_the_midpoint(self):
        # Source and detector are interchangeable -- reciprocity. If this ever
        # failed, the two fluence terms would not be describing the same
        # medium.
        separation = 0.4
        x = np.linspace(-0.2, 0.6, 81)
        z = np.linspace(0.005, 0.4, 61)
        kernel = SNOW.sensitivity_kernel(separation, x, z)
        mirrored = SNOW.sensitivity_kernel(separation, separation - x[::-1], z)
        assert np.allclose(kernel, mirrored[:, ::-1], rtol=1e-9)

    def test_the_kernel_is_largest_at_the_source_and_detector(self):
        """The absolute maximum is at the ends, not in the middle.

        Worth pinning because the name misleads. Fluence diverges as ``1/r``
        at a point source, so the product peaks where the source and detector
        sit. The banana is the shape of the *contours* between them -- the
        region still carrying appreciable weight -- not the location of the
        maximum. A figure drawn on a linear scale shows two bright spots and
        no banana at all.
        """
        separation = 0.4
        x = np.linspace(-0.2, 0.6, 81)
        z = np.linspace(0.005, 0.4, 61)
        kernel = SNOW.sensitivity_kernel(separation, x, z)
        peak_x = x[int(np.unravel_index(kernel.argmax(), kernel.shape)[1])]
        assert np.isclose(peak_x, 0.0, atol=0.02) or np.isclose(
            peak_x, separation, atol=0.02
        )

    def test_the_kernel_is_normalised(self):
        kernel = SNOW.sensitivity_kernel(
            0.3, np.linspace(-0.1, 0.4, 40), np.linspace(0.005, 0.3, 40)
        )
        assert kernel.max() == pytest.approx(1.0)
        assert np.all(kernel >= 0)

    def test_an_object_outside_the_banana_is_invisible(self):
        # The whole reason geometry matters: sensitivity at four times the
        # probing depth is orders of magnitude down, so an object there cannot
        # be detected however strongly it absorbs.
        separation = 0.4
        x = np.array([separation / 2])
        peak_depth = SNOW.probing_depth(separation)
        z = np.array([peak_depth, 4 * peak_depth])
        kernel = SNOW.sensitivity_kernel(separation, x, z)
        assert kernel[1, 0] < 0.05 * kernel[0, 0]

    def test_rejects_a_nonpositive_separation(self):
        with pytest.raises(ValueError, match="must be positive"):
            SNOW.probing_depth(0.0)


class TestTwoWayDepth:
    def test_matches_the_closed_form(self):
        assert two_way_detection_depth(0.10, 0.01) == pytest.approx(
            0.05 * np.log(100)
        )

    def test_a_stricter_threshold_gives_a_shallower_depth(self):
        assert two_way_detection_depth(0.1, 0.001) > two_way_detection_depth(
            0.1, 0.01
        )

    def test_scales_with_penetration_depth(self):
        assert two_way_detection_depth(0.2) == pytest.approx(
            2 * two_way_detection_depth(0.1)
        )

    def test_rejects_an_impossible_signal_fraction(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            two_way_detection_depth(0.1, 1.5)


class TestValidation:
    def test_rejects_negative_absorption(self):
        with pytest.raises(ValueError, match="non-negative"):
            DiffusionParameters(-1.0, 548.0)

    def test_rejects_a_refractive_index_below_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            DiffusionParameters(1e-3, 548.0, refractive_index=0.9)
