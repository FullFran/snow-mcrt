"""The mechanism behind grain-size sensitivity, checked rather than asserted.

`docs/remote-sensing.md` explains why the visible band cannot size a grain and
the shortwave infrared can, in terms of two competing scalings that cancel in
one regime and not in the other. These tests hold that explanation to the
numbers, so the document cannot quietly drift away from the physics it claims.

References:
    Wiscombe, W. J., & Warren, S. G. (1980). A model for the spectral albedo of
        snow. I: Pure snow. *Journal of the Atmospheric Sciences*, 37, 2712-2733.
    Warren, S. G., & Brandt, R. E. (2008). *Journal of Geophysical Research*,
        113, D14220.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants
from snow_mcrt.domain.medium import SnowLayer, compute_layer_properties

ICE = Path(__file__).resolve().parents[1] / "data" / "ice" / "warren_brandt_2008.dat"

RADII_UM = (50.0, 100.0, 250.0, 500.0, 1000.0)
VISIBLE_NM = 560.0
SHORTWAVE_NM = 1610.0


@pytest.fixture(scope="module")
def constants():
    return TabulatedConstants(ICE, wavelength_scale_to_nm=1000.0).load()


@pytest.fixture(scope="module")
def solver():
    return MiepythonSolver()


def _absorption_lengths(constants, solver, wavelength_nm: float) -> list[float]:
    """1 / (sigma_ext * (1 - omega)) at each radius: how far light gets."""
    m = constants.m_at(np.array([wavelength_nm]))
    lengths = []
    for radius_um in RADII_UM:
        layer = SnowLayer(grain_radius_m=radius_um * 1e-6, density=300.0)
        p = compute_layer_properties(solver, layer, m, np.array([wavelength_nm]))
        co_albedo = 1.0 - p.single_scattering_albedo[0]
        lengths.append(1.0 / (p.extinction_coefficient[0] * co_albedo))
    return lengths


class TestTheTwoScalings:
    def test_co_albedo_grows_with_grain_size(self, constants, solver):
        """Absorption within a grain scales with the path through it."""
        for wavelength in (VISIBLE_NM, SHORTWAVE_NM):
            m = constants.m_at(np.array([wavelength]))
            values = []
            for radius_um in RADII_UM:
                layer = SnowLayer(grain_radius_m=radius_um * 1e-6, density=300.0)
                p = compute_layer_properties(solver, layer, m, np.array([wavelength]))
                values.append(1.0 - p.single_scattering_albedo[0])
            assert values == sorted(values), (
                f"co-albedo must increase with radius at {wavelength} nm"
            )

    def test_extinction_falls_with_grain_size(self, constants, solver):
        """At fixed density, larger grains means fewer of them."""
        m = constants.m_at(np.array([VISIBLE_NM]))
        values = []
        for radius_um in RADII_UM:
            layer = SnowLayer(grain_radius_m=radius_um * 1e-6, density=300.0)
            p = compute_layer_properties(solver, layer, m, np.array([VISIBLE_NM]))
            values.append(p.extinction_coefficient[0])
        assert values == sorted(values, reverse=True)

    def test_extinction_scales_roughly_as_the_inverse_radius(self, constants, solver):
        """The geometric-optics expectation, to within a few per cent."""
        m = constants.m_at(np.array([VISIBLE_NM]))

        def sigma(radius_um: float) -> float:
            layer = SnowLayer(grain_radius_m=radius_um * 1e-6, density=300.0)
            return compute_layer_properties(
                solver, layer, m, np.array([VISIBLE_NM])
            ).extinction_coefficient[0]

        # Twenty-fold in radius should be twenty-fold down in extinction.
        ratio = sigma(50.0) / sigma(1000.0)
        assert ratio == pytest.approx(20.0, rel=0.05)


class TestTheCancellation:
    def test_in_the_visible_the_two_scalings_cancel(self, constants, solver):
        """The absorption length is nearly invariant across a factor of twenty
        in radius. This is why the visible band cannot size a grain."""
        lengths = _absorption_lengths(constants, solver, VISIBLE_NM)

        spread = max(lengths) / min(lengths)
        assert spread < 1.1, (
            f"absorption length varies by {spread:.2f}x in the visible; the "
            "cancellation that makes grain size invisible there has broken"
        )
        assert all(30.0 < length < 45.0 for length in lengths)

    def test_in_the_shortwave_infrared_the_cancellation_breaks(self, constants, solver):
        """The absorption length grows several-fold with grain size. This is
        what the shortwave band reads."""
        lengths = _absorption_lengths(constants, solver, SHORTWAVE_NM)

        assert lengths == sorted(lengths)
        assert max(lengths) / min(lengths) > 3.0

    def test_the_two_regimes_differ_by_orders_of_magnitude(self, constants, solver):
        """Metres in the visible, millimetres in the shortwave infrared."""
        visible = _absorption_lengths(constants, solver, VISIBLE_NM)
        shortwave = _absorption_lengths(constants, solver, SHORTWAVE_NM)

        assert min(visible) / max(shortwave) > 1000.0


class TestTheDocumentedNumbers:
    """The figures quoted in docs/remote-sensing.md, so prose and code cannot
    drift apart."""

    def test_visible_absorption_length_is_around_thirty_eight_metres(
        self, constants, solver
    ):
        lengths = _absorption_lengths(constants, solver, VISIBLE_NM)
        assert lengths[0] == pytest.approx(35.95, abs=0.5)
        assert lengths[-1] == pytest.approx(38.07, abs=0.5)

    def test_shortwave_absorption_length_runs_from_millimetres_to_millimetres(
        self, constants, solver
    ):
        lengths = _absorption_lengths(constants, solver, SHORTWAVE_NM)
        assert lengths[0] == pytest.approx(0.0013, abs=2e-4)
        assert lengths[-1] == pytest.approx(0.0066, abs=5e-4)

    def test_ice_absorption_rises_by_orders_of_magnitude_between_the_two_bands(
        self, constants
    ):
        """The single fact the whole signature rests on."""
        k_visible = constants.m_at(np.array([VISIBLE_NM]))[0].imag
        k_shortwave = constants.m_at(np.array([SHORTWAVE_NM]))[0].imag
        assert k_shortwave / k_visible > 1e4
