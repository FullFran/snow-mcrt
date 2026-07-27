"""Conventions: units, sign, and the refusal to extrapolate."""

import numpy as np
import pytest

from snow_mcrt.domain.optics import OpticalConstants, size_parameter


def constants(**overrides):
    kwargs = dict(
        wavelength_nm=np.array([400.0, 500.0, 600.0]),
        n=np.array([1.32, 1.31, 1.31]),
        k=np.array([1.0e-9, 2.0e-9, 3.0e-9]),
        name="synthetic",
    )
    kwargs.update(overrides)
    return OpticalConstants(**kwargs)


class TestConstruction:
    def test_rejects_descending_wavelengths(self):
        with pytest.raises(ValueError, match="ascending"):
            constants(wavelength_nm=np.array([600.0, 500.0, 400.0]))

    def test_rejects_negative_k_as_gain(self):
        with pytest.raises(ValueError, match="gain"):
            constants(k=np.array([1e-9, -1e-9, 3e-9]))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shapes disagree"):
            constants(n=np.array([1.31, 1.31]))

    def test_transparent_medium_is_legitimate(self):
        c = constants(k=np.zeros(3))
        assert np.all(c.absorption_coefficient([450.0, 550.0]) == 0.0)


class TestInterpolation:
    def test_returns_tabulated_value_at_a_grid_point(self):
        m = constants().m_at(500.0)
        assert m.real == pytest.approx(1.31)
        assert m.imag == pytest.approx(2.0e-9)

    def test_interpolates_linearly_between_grid_points(self):
        m = constants().m_at(450.0)
        assert m.real == pytest.approx(1.315)
        assert m.imag == pytest.approx(1.5e-9)

    def test_refuses_to_extrapolate_below_the_table(self):
        with pytest.raises(ValueError, match="outside the tabulated range"):
            constants().m_at(300.0)

    def test_refuses_to_extrapolate_above_the_table(self):
        with pytest.raises(ValueError, match="outside the tabulated range"):
            constants().m_at([500.0, 900.0])


class TestAbsorptionCoefficient:
    def test_matches_the_closed_form(self):
        c = constants()
        gamma = c.absorption_coefficient(500.0)
        expected = 4.0 * np.pi * 2.0e-9 / 500.0e-9
        assert gamma[0] == pytest.approx(expected)

    def test_is_expressed_in_inverse_metres(self):
        # 4*pi*k/lambda with k=2e-9 at 500 nm is of order 0.05 m^-1, i.e. an
        # e-folding length of ~20 m in solid bubble-free ice. A result in
        # inverse nanometres would be off by 1e9 and is what this pins.
        gamma = constants().absorption_coefficient(500.0)[0]
        assert 1e-3 < gamma < 1e3


class TestSizeParameter:
    def test_matches_the_closed_form(self):
        x = size_parameter(radius_m=100e-6, wavelength_nm=500.0)
        assert x[0] == pytest.approx(2 * np.pi * 100e-6 / 500e-9)

    def test_snow_grains_are_deep_in_the_geometric_regime(self):
        # A 100 um grain in the visible: x of order 1e3. Any implementation
        # returning O(1) has confused metres with micrometres.
        x = size_parameter(radius_m=100e-6, wavelength_nm=500.0)[0]
        assert 1e2 < x < 1e4

    def test_broadcasts_radii_against_wavelengths(self):
        x = size_parameter(
            radius_m=np.array([[50e-6], [100e-6]]),
            wavelength_nm=np.array([400.0, 500.0, 600.0]),
        )
        assert x.shape == (2, 3)

    def test_scales_inversely_with_wavelength(self):
        x = size_parameter(100e-6, np.array([400.0, 800.0]))
        assert x[0] == pytest.approx(2 * x[1])

    def test_rejects_nonpositive_wavelength(self):
        with pytest.raises(ValueError, match="wavelength must be positive"):
            size_parameter(100e-6, 0.0)
