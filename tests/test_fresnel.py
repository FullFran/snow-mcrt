"""Ground truth for the interface: closed forms Fresnel must reproduce.

Every test here compares against something known independently of the
implementation -- normal-incidence reflectance, Brewster's angle, the critical
angle, Snell's law, and the reciprocity between entering and leaving a medium.
An interface that satisfies all of them is constrained everywhere it matters.

This is the piece the 1-D transport never had. A photon reaching `z = 0` in a
plane-parallel run simply left. Snow at `n = 1.31` reflects a substantial part
of the diffuse flux arriving at the surface back into the pack, which is why
the diffusion solution needs an extrapolated boundary well outside the
surface -- it is not a correction that can be dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.fresnel import (
    critical_angle,
    fresnel_reflectance,
    refract,
    specular_reflect,
)

ICE = 1.31
AIR = 1.0


def schlick(n1: float, n2: float, cos_i: float) -> float:
    """Schlick's approximation -- close to Fresnel away from grazing."""
    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_i) ** 5


class TestNormalIncidence:
    def test_matches_the_closed_form(self):
        expected = ((ICE - AIR) / (ICE + AIR)) ** 2
        assert fresnel_reflectance(1.0, ICE, AIR) == pytest.approx(expected)

    def test_is_the_same_entering_or_leaving(self):
        # Reflectance at normal incidence does not care which side you are on.
        assert fresnel_reflectance(1.0, ICE, AIR) == pytest.approx(
            fresnel_reflectance(1.0, AIR, ICE)
        )

    def test_a_matched_boundary_reflects_nothing(self):
        assert fresnel_reflectance(0.4, 1.31, 1.31) == pytest.approx(0.0)

    def test_snow_reflects_about_1_8_percent_at_normal_incidence(self):
        # The value a snow-optics reader should recognise: n = 1.31 is a weak
        # interface head-on, which is exactly why the *diffuse* internal
        # reflection being near 0.5 is surprising and worth stating.
        assert fresnel_reflectance(1.0, ICE, AIR) == pytest.approx(0.0180, abs=5e-4)


class TestGrazingAndCritical:
    def test_grazing_incidence_reflects_everything(self):
        assert fresnel_reflectance(1e-9, AIR, ICE) == pytest.approx(1.0, abs=1e-6)

    def test_beyond_the_critical_angle_is_total(self):
        # Leaving the denser medium. Past the critical angle there is no
        # transmitted direction at all, so R is exactly 1 -- not approximately.
        theta_c = critical_angle(ICE, AIR)
        for theta in (theta_c + 1e-6, theta_c + 0.2, np.pi / 2 - 1e-9):
            assert fresnel_reflectance(np.cos(theta), ICE, AIR) == 1.0

    def test_just_inside_the_critical_angle_is_not_total(self):
        theta_c = critical_angle(ICE, AIR)
        assert fresnel_reflectance(np.cos(theta_c - 1e-3), ICE, AIR) < 1.0

    def test_the_critical_angle_matches_snells_law(self):
        assert critical_angle(ICE, AIR) == pytest.approx(np.arcsin(AIR / ICE))

    def test_there_is_no_critical_angle_going_into_a_denser_medium(self):
        assert np.isnan(critical_angle(AIR, ICE))


class TestAgainstIndependentForms:
    @pytest.mark.parametrize("cos_i", [1.0, 0.95, 0.8, 0.6])
    def test_tracks_schlick_away_from_grazing(self, cos_i):
        # Schlick is an approximation, not an oracle -- but it agrees with
        # Fresnel to about a percent of reflectance over this range, which is
        # enough to catch a swapped index or a dropped polarisation term.
        assert fresnel_reflectance(cos_i, AIR, ICE) == pytest.approx(
            schlick(AIR, ICE, cos_i), abs=0.01
        )

    def test_reflectance_rises_monotonically_towards_grazing(self):
        cos_i = np.linspace(1.0, 1e-6, 60)
        r = fresnel_reflectance(cos_i, AIR, ICE)
        assert np.all(np.diff(r) > 0)

    def test_stays_within_zero_and_one(self):
        cos_i = np.linspace(1e-9, 1.0, 200)
        for n1, n2 in ((AIR, ICE), (ICE, AIR)):
            r = fresnel_reflectance(cos_i, n1, n2)
            assert np.all((r >= 0.0) & (r <= 1.0))


class TestSpecularReflection:
    def test_flips_only_the_component_along_the_normal(self):
        direction = np.array([[0.6, 0.0, 0.8]])
        normal = np.array([[0.0, 0.0, -1.0]])
        out = specular_reflect(direction, normal)
        np.testing.assert_allclose(out, [[0.6, 0.0, -0.8]], atol=1e-12)

    def test_preserves_the_unit_length(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(64, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        n = np.tile([0.0, 0.0, -1.0], (64, 1))
        out = specular_reflect(d, n)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)

    def test_reflecting_twice_returns_the_original(self):
        d = np.array([[0.3, -0.4, 0.8660254037844387]])
        n = np.array([[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(
            specular_reflect(specular_reflect(d, n), n), d, atol=1e-12
        )


class TestRefraction:
    """``z`` is depth, positive downward, so a photon leaving through the
    surface has ``u_z < 0`` and the normal oriented against it is ``+z``."""

    SURFACE_NORMAL = np.array([[0.0, 0.0, 1.0]])

    def test_obeys_snells_law(self):
        theta_i = 0.5
        d = np.array([[np.sin(theta_i), 0.0, -np.cos(theta_i)]])
        out = refract(d, self.SURFACE_NORMAL, ICE, AIR)
        theta_t = np.arcsin(np.clip(out[0, 0], -1, 1))
        assert ICE * np.sin(theta_i) == pytest.approx(AIR * np.sin(theta_t))

    def test_normal_incidence_passes_straight_through(self):
        d = np.array([[0.0, 0.0, -1.0]])
        np.testing.assert_allclose(
            refract(d, self.SURFACE_NORMAL, ICE, AIR), d, atol=1e-12
        )

    def test_preserves_the_unit_length(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=(128, 3))
        d[:, 2] = -np.abs(d[:, 2])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        n = np.tile(self.SURFACE_NORMAL, (128, 1))
        # Only where refraction is defined; past the critical angle the caller
        # must have reflected instead.
        cos_i = -np.sum(d * n, axis=1)
        ok = cos_i > np.cos(critical_angle(ICE, AIR))
        assert ok.any()
        out = refract(d[ok], n[ok], ICE, AIR)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)

    def test_bends_away_from_the_normal_when_leaving_the_denser_medium(self):
        theta_i = 0.4
        d = np.array([[np.sin(theta_i), 0.0, -np.cos(theta_i)]])
        out = refract(d, self.SURFACE_NORMAL, ICE, AIR)
        assert np.arcsin(out[0, 0]) > theta_i

    def test_a_matched_boundary_does_not_bend(self):
        d = np.array([[0.5, 0.1, -0.86]])
        d = d / np.linalg.norm(d)
        np.testing.assert_allclose(
            refract(d, self.SURFACE_NORMAL, ICE, ICE), d, atol=1e-12
        )

    def test_the_transmitted_ray_stays_on_the_far_side(self):
        # A refracted photon must keep going outward. A sign slip in the
        # vector form sends it back into the medium, which is silent: the
        # photon simply scatters again and the reflectance comes out low.
        rng = np.random.default_rng(2)
        d = rng.normal(size=(256, 3))
        d[:, 2] = -np.abs(d[:, 2])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        n = np.tile(self.SURFACE_NORMAL, (256, 1))
        ok = -np.sum(d * n, axis=1) > np.cos(critical_angle(ICE, AIR))
        assert np.all(refract(d[ok], n[ok], ICE, AIR)[:, 2] < 0)


class TestAgreesWithTheDiffusionBoundary:
    """The bridge between this module and :mod:`~snow_mcrt.domain.diffusion`.

    Diffusion theory does not trace rays; it carries the boundary as a single
    effective internal reflection coefficient from an empirical fit. Fresnel
    integrated over the angular distribution that actually arrives there is a
    completely independent route to the same number. If the two disagree, a
    3-D run and the diffusion solution it is validated against are describing
    different surfaces, and every comparison downstream is meaningless.
    """

    def test_the_diffuse_internal_reflection_matches_the_empirical_fit(self):
        theta = np.linspace(0.0, np.pi / 2, 200_001)
        # Lambertian interior: the flux arriving at the boundary per unit
        # angle goes as 2 sin(theta) cos(theta).
        weight = 2.0 * np.sin(theta) * np.cos(theta)
        integrated = np.trapezoid(
            fresnel_reflectance(np.cos(theta), ICE, AIR) * weight, theta
        )
        fitted = DiffusionParameters(
            absorption_coefficient=1.0,
            reduced_scattering_coefficient=100.0,
            refractive_index=ICE,
        ).internal_reflection
        assert integrated == pytest.approx(fitted, abs=2e-3)

    def test_most_of_the_arriving_flux_is_beyond_the_critical_angle(self):
        # 58% of it, which is why the diffuse reflection is 0.45 while normal
        # incidence is 0.018. Dropping the interface is not a small error.
        beyond = 1.0 - np.cos(critical_angle(ICE, AIR)) ** 2
        assert beyond == pytest.approx(0.583, abs=0.005)


class TestVectorisation:
    def test_reflectance_broadcasts_over_an_array(self):
        cos_i = np.linspace(0.1, 1.0, 17)
        assert fresnel_reflectance(cos_i, AIR, ICE).shape == (17,)

    def test_a_scalar_cosine_gives_a_scalar(self):
        assert np.ndim(fresnel_reflectance(0.5, AIR, ICE)) == 0
