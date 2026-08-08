"""Band integration: does the instrument step behave like an instrument."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from snow_mcrt.application.band_albedo import (
    integrate_bands,
    response_shape_sensitivity,
)
from snow_mcrt.domain.sensor import (
    MODIS_LAND,
    SENTINEL2_MSI,
    Band,
    Gaussian,
    Tabulated,
    TopHat,
    band_albedo,
    ndsi,
)

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference"


def _curve(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A committed reference albedo curve, as (wavelength, albedo)."""
    with open(REFERENCE / f"{name}.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.array([float(r["wavelength_nm"]) for r in rows]),
        np.array([float(r["albedo"]) for r in rows]),
    )


class TestIntegration:
    def test_a_constant_albedo_integrates_to_itself(self):
        """The weight cancels: this is the identity the formula must satisfy."""
        grid = np.linspace(300.0, 2500.0, 400)
        flat = np.full_like(grid, 0.7)
        for band in SENTINEL2_MSI.values():
            assert band_albedo(grid, flat, band) == pytest.approx(0.7, abs=1e-12)

    def test_a_narrow_band_approaches_the_albedo_at_its_centre(self):
        """As the band closes on a point, the integral must return the point."""
        grid = np.linspace(400.0, 800.0, 4001)
        albedo = 0.9 - 1e-4 * (grid - 400.0)
        centre = 600.0
        at_centre = float(np.interp(centre, grid, albedo))

        narrow = band_albedo(grid, albedo, Band("n", TopHat(centre, 1.0)))
        wide = band_albedo(grid, albedo, Band("w", TopHat(centre, 200.0)))

        assert narrow == pytest.approx(at_centre, abs=1e-4)
        # The wide band still averages a linear ramp symmetrically, so it lands
        # on the centre too -- what must differ is that it is not an identity.
        assert abs(wide - at_centre) < 1e-3

    def test_a_sloping_source_pulls_the_answer_towards_the_bright_end(self):
        """Irradiance weighting is not decoration: it moves the number."""
        grid = np.linspace(500.0, 700.0, 2001)
        albedo = np.linspace(0.2, 0.8, grid.size)
        band = Band("wide", TopHat(600.0, 180.0))

        flat = band_albedo(grid, albedo, band)
        red_heavy = band_albedo(grid, albedo, band, irradiance=grid**4)

        assert red_heavy > flat

    def test_a_band_outside_the_grid_raises(self):
        """Silence here would let an unmeasured band travel as if measured."""
        grid = np.linspace(400.0, 900.0, 100)
        albedo = np.full_like(grid, 0.8)
        with pytest.raises(ValueError, match="no overlap"):
            band_albedo(grid, albedo, SENTINEL2_MSI["B11"])

    def test_a_tabulated_response_reproduces_the_top_hat_it_describes(self):
        """The adapter for real instrument curves must agree on a shape it can express."""
        grid = np.linspace(300.0, 2500.0, 2201)
        albedo = 0.9 - 2e-4 * (grid - 300.0)

        edges = np.array([539.0, 540.0, 580.0, 581.0])
        tabulated = Band("T", Tabulated(560.0, edges, np.array([0.0, 1.0, 1.0, 0.0])))
        top_hat = Band("H", TopHat(560.0, 40.0))

        assert band_albedo(grid, albedo, tabulated) == pytest.approx(
            band_albedo(grid, albedo, top_hat), abs=1e-4
        )


class TestSnowSignature:
    """The physics the index rests on, checked on committed curves."""

    def test_clean_snow_is_bright_in_the_visible_and_dark_in_the_swir(self):
        grid, albedo = _curve("pure-r100um")
        result = integrate_bands(grid, albedo, SENTINEL2_MSI, "sentinel2")

        assert result.band_albedo["B3"] > 0.9
        assert result.band_albedo["B11"] < 0.3
        assert result.band_albedo["B3"] > 3 * result.band_albedo["B11"]

    def test_clean_snow_scores_far_above_the_operational_ndsi_threshold(self):
        """Hall et al. (1995) map snow at NDSI > 0.4. Clean snow must clear it."""
        grid, albedo = _curve("pure-r100um")
        for bands, instrument in (
            (SENTINEL2_MSI, "sentinel2"),
            (MODIS_LAND, "modis"),
        ):
            result = integrate_bands(grid, albedo, bands, instrument)
            assert result.ndsi is not None
            assert result.ndsi > 0.4

    def test_the_two_instruments_agree_on_the_index_without_being_identical(self):
        """Different bands, same physics: close, but not the same number."""
        grid, albedo = _curve("pure-r100um")
        s2 = integrate_bands(grid, albedo, SENTINEL2_MSI, "sentinel2").ndsi
        modis = integrate_bands(grid, albedo, MODIS_LAND, "modis").ndsi

        assert abs(s2 - modis) < 0.1
        assert s2 != modis

    @pytest.mark.parametrize(
        "coarser, finer",
        [("pure-r1000um", "pure-r100um"), ("pure-r250um", "pure-r50um")],
    )
    def test_coarser_grains_are_darker_in_the_shortwave_infrared(self, coarser, finer):
        """Grain size is read in the SWIR, which is why NDSI cannot retrieve it."""
        grid_c, albedo_c = _curve(coarser)
        grid_f, albedo_f = _curve(finer)

        swir_c = band_albedo(grid_c, albedo_c, SENTINEL2_MSI["B11"])
        swir_f = band_albedo(grid_f, albedo_f, SENTINEL2_MSI["B11"])
        assert swir_c < swir_f

    def test_grain_size_barely_moves_the_visible_band(self):
        """The other half of why NDSI detects snow but does not size it."""
        grid_c, albedo_c = _curve("pure-r1000um")
        grid_f, albedo_f = _curve("pure-r50um")

        green_c = band_albedo(grid_c, albedo_c, SENTINEL2_MSI["B3"])
        green_f = band_albedo(grid_f, albedo_f, SENTINEL2_MSI["B3"])
        assert abs(green_c - green_f) < 0.05

    def test_black_carbon_darkens_the_visible_and_spares_the_swir(self):
        """Impurities and grain size act on different bands. That separability
        is what makes a two-parameter retrieval possible at all."""
        grid_clean, clean = _curve("bc0ngg-r100um")
        grid_dirty, dirty = _curve("bc1000ngg-r100um")

        green_drop = band_albedo(grid_clean, clean, SENTINEL2_MSI["B3"]) - band_albedo(
            grid_dirty, dirty, SENTINEL2_MSI["B3"]
        )
        swir_drop = band_albedo(grid_clean, clean, SENTINEL2_MSI["B11"]) - band_albedo(
            grid_dirty, dirty, SENTINEL2_MSI["B11"]
        )

        assert green_drop > 0.05
        assert green_drop > swir_drop


class TestResponseShape:
    def test_the_top_hat_approximation_is_quantified_not_asserted(self):
        grid, albedo = _curve("pure-r100um")
        differences = response_shape_sensitivity(grid, albedo, SENTINEL2_MSI)

        assert set(differences) == set(SENTINEL2_MSI)
        # Every band must stay within a few per cent, or the idealisation is
        # not fit to be shipped as the default.
        assert max(differences.values()) < 0.05

    def test_the_shape_matters_least_where_the_curve_is_flat(self):
        """Visible bands sit on a flat albedo; SWIR bands sit on a slope."""
        grid, albedo = _curve("pure-r100um")
        differences = response_shape_sensitivity(grid, albedo, SENTINEL2_MSI)

        assert differences["B3"] < differences["B12"]


class TestIndex:
    def test_it_is_symmetric_under_swapping_the_bands(self):
        assert ndsi(0.9, 0.1) == pytest.approx(-ndsi(0.1, 0.9))

    def test_equal_bands_give_zero(self):
        assert ndsi(0.5, 0.5) == 0.0

    def test_two_dark_bands_are_undefined_rather_than_zero(self):
        with pytest.raises(ValueError):
            ndsi(0.0, 0.0)


class TestGaussian:
    def test_it_is_normalised_at_its_centre_and_half_at_the_half_width(self):
        response = Gaussian(600.0, 40.0)
        assert response(np.array([600.0]))[0] == pytest.approx(1.0)
        assert response(np.array([620.0]))[0] == pytest.approx(0.5, abs=1e-9)

    def test_it_is_truncated_rather_than_left_with_infinite_tails(self):
        response = Gaussian(600.0, 40.0)
        assert response(np.array([900.0]))[0] == 0.0
