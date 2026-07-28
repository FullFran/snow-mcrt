"""The comparison must be a measurement of the approximation, not of a setup.

These tests deliberately run few photons. The *number* -- how far diffusion
departs from transport at a given separation -- belongs in a committed CSV
produced by a run script at a photon count that can resolve the tail. What
belongs here is everything that would make such a number meaningless: a
mismatched surface, a mismatched source, bins in the wrong place, or a ratio
computed against empty bins.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.application.validate_diffusion import compare_with_diffusion
from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.transport import TransportConfig

# Alpine snow near 450 nm: high albedo, strongly forward scattering.
OMEGA, G, BETA = 0.99, 0.85, 1000.0


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def comparison(backend):
    return compare_with_diffusion(
        backend,
        OMEGA,
        G,
        BETA,
        config=TransportConfig(n_photons=20000, seed=5, max_scatters=4000),
        n_bins=16,
    )


class TestBothSolversSeeTheSameProblem:
    def test_the_diffusion_side_uses_the_same_refractive_index(self, backend):
        # The single most damaging way to get this wrong. Diffusion carries
        # the index as an internal reflection coefficient; if the comparison
        # left it at a default while the engine ran at 1.31, the ratio would
        # be a measurement of two different surfaces.
        result = compare_with_diffusion(
            backend,
            OMEGA,
            G,
            BETA,
            config=TransportConfig(n_photons=800, seed=6, max_scatters=500),
            surface_index=1.4,
            n_bins=8,
        )
        expected = DiffusionParameters.from_optical_properties(
            OMEGA, G, BETA, refractive_index=1.4
        )
        np.testing.assert_allclose(
            result.diffusion,
            expected.diffuse_reflectance(result.rho_m),
            rtol=1e-12,
        )
        assert result.surface_index == 1.4

    def test_the_transport_length_matches_the_diffusion_parameters(
        self, comparison
    ):
        expected = DiffusionParameters.from_optical_properties(
            OMEGA, G, BETA, refractive_index=1.31
        )
        assert comparison.transport_mfp_m == pytest.approx(
            expected.transport_mean_free_path
        )
        assert comparison.penetration_depth_m == pytest.approx(
            expected.penetration_depth
        )


class TestTheBinsSitWhereDiffusionLives:
    def test_the_innermost_bin_is_at_least_one_transport_length_out(
        self, comparison
    ):
        # Inside one transport mean free path diffusion is not inaccurate,
        # it is undefined: the photon has not yet forgotten which way it was
        # going, which is the entire premise.
        assert comparison.rho_in_mfp[0] >= 1.0

    def test_the_range_is_reported_in_transport_lengths(self, comparison):
        # A validity range quoted in centimetres does not transfer to another
        # snowpack. Quoted in mfp' it does.
        np.testing.assert_allclose(
            comparison.rho_in_mfp,
            comparison.rho_m / comparison.transport_mfp_m,
        )

    def test_the_outer_edge_reaches_well_past_the_penetration_depth(
        self, comparison
    ):
        assert comparison.rho_m[-1] > 10 * comparison.penetration_depth_m


class TestTheRatio:
    def test_is_not_computed_against_empty_bins(self, backend):
        # The tail runs out of photons before it runs out of bins, and an
        # empty bin is not a measurement of zero. Those bins must be excluded
        # rather than reported as a ratio of zero.
        result = compare_with_diffusion(
            backend,
            OMEGA,
            G,
            BETA,
            config=TransportConfig(n_photons=400, seed=7, max_scatters=400),
            n_bins=30,
        )
        assert not result.sampled().all()
        assert np.all(result.monte_carlo[result.sampled()] > 0)

    def test_is_one_where_the_two_agree_exactly(self, comparison):
        # A self-consistency check on the ratio itself, independent of any
        # physics: feeding the diffusion curve in as the Monte Carlo gives
        # exactly one.
        import dataclasses

        identical = dataclasses.replace(
            comparison, monte_carlo=comparison.diffusion.copy()
        )
        np.testing.assert_allclose(identical.ratio[identical.sampled()], 1.0)

    def test_agrees_with_diffusion_to_within_a_factor_of_two_nearby(
        self, comparison
    ):
        # Deliberately loose. At this photon count the assertion is that the
        # two solvers describe the same physical situation at all -- a
        # swapped source, a dropped interface or a units slip would miss by
        # orders of magnitude, not by tens of percent.
        assert comparison.worst_ratio_within(12.0) < 1.0

    def test_diffusion_overestimates_right_at_the_source(self, backend):
        # Measured on a P100 at two million photons, the innermost bin comes
        # back at 0.79 to 0.80 in all four snowpacks tested -- consistent to
        # a percent across three orders of magnitude in co-albedo. Inside a
        # few transport lengths the photon has not yet forgotten which way it
        # was going, so diffusion is not inaccurate there, it is inapplicable.
        result = compare_with_diffusion(
            backend,
            OMEGA,
            G,
            BETA,
            config=TransportConfig(n_photons=200_000, seed=8, max_scatters=6000),
            n_bins=20,
        )
        innermost = result.ratio[result.sampled()][0]
        assert innermost < 0.95

    def test_the_useful_band_agrees_far_better_than_the_worst_case(
        self, backend
    ):
        # The reason departure_between exists. A worst case taken over
        # everything inside twelve transport lengths is dominated by the near
        # field and lands near 20%; the band a source-detector pair actually
        # uses is several times tighter. Reporting only the first would
        # describe the premise failing rather than the approximation
        # degrading.
        result = compare_with_diffusion(
            backend,
            OMEGA,
            G,
            BETA,
            config=TransportConfig(n_photons=200_000, seed=9, max_scatters=6000),
            n_bins=20,
        )
        assert result.departure_between(3.0, 12.0) < result.worst_ratio_within(12.0)

    def test_a_band_with_no_bins_in_it_is_not_invented(self, comparison):
        assert np.isnan(comparison.departure_between(1e6, 1e7))


class TestTheRunIsUsable:
    def test_reports_what_was_left_in_flight(self, comparison):
        # An unconverged tail and a genuinely small tail look identical in a
        # plot. Only this number tells them apart.
        assert comparison.truncated >= 0.0

    def test_carries_every_parameter_that_produced_it(self, comparison):
        assert comparison.single_scattering_albedo == OMEGA
        assert comparison.asymmetry == G
        assert comparison.extinction_coefficient == BETA
        assert comparison.n_photons == 20000

    def test_columns_are_all_the_same_length(self, comparison):
        lengths = {len(v) for v in comparison.columns().values()}
        assert lengths == {comparison.rho_m.size}
