"""The 3-D engine, checked against what is known without it.

Three kinds of oracle appear here, in increasing order of how much they
constrain:

1. **Exact invariants.** The energy ledger, unit-length directions, and the
   rotation branch at ``|u_z| = 1``. These hold photon by photon and are
   checked to floating-point, not to Monte Carlo noise.
2. **The 1-D engine.** With a matched boundary and the same phase function,
   the two engines are solving the same problem by different bookkeeping.
   Total reflectance must agree within the noise of both. This is the
   strongest test in the file: the 1-D engine is already validated against van
   de Hulst, so agreement transfers that validation to three dimensions.
3. **Fresnel.** Turning the real interface on must move the answer in a
   direction that can be argued for in advance.

The radial profile is *not* checked against diffusion here. That comparison
belongs in the application layer, where it can be run at the photon counts it
needs and reported with a residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.domain.transport import TransportConfig, run_transport
from snow_mcrt.domain.transport3d import (
    log_radial_edges,
    run_transport_3d,
)


@pytest.fixture
def backend():
    return NumpyBackend()


def edges():
    return log_radial_edges(1e-3, 1.0, 24)


class TestExactInvariants:
    def test_the_energy_ledger_balances_without_roulette(self, backend):
        result = run_transport_3d(
            backend,
            extinction_coefficient=100.0,
            single_scattering_albedo=0.9,
            asymmetry=0.5,
            config=TransportConfig(n_photons=4000, seed=3, roulette_threshold=0.0),
        )
        assert result.energy_balance == pytest.approx(0.0, abs=1e-12)

    def test_the_ledger_balances_in_expectation_with_roulette(self, backend):
        """Roulette conserves energy in expectation, not per realisation.

        Killed photons take their weight with them and survivors are boosted
        by ``1/p`` to compensate, so the residual is of order the roulette
        threshold and changes sign between runs. The bound is set by that
        scale, exactly as in the plane-parallel engine -- demanding
        exactness would be demanding that an unbiased estimator also be a
        conserved quantity.
        """
        config = TransportConfig(n_photons=4000, seed=4)
        result = run_transport_3d(
            backend,
            extinction_coefficient=100.0,
            single_scattering_albedo=0.9,
            asymmetry=0.5,
            config=config,
        )
        assert abs(result.energy_balance) < 100 * config.roulette_threshold

    def test_every_channel_is_non_negative(self, backend):
        result = run_transport_3d(
            backend, 100.0, 0.8, 0.3, config=TransportConfig(n_photons=2000, seed=5)
        )
        assert min(result.reflected, result.absorbed, result.truncated) >= 0.0

    def test_directions_stay_unit_vectors(self, backend):
        result = run_transport_3d(
            backend,
            100.0,
            0.95,
            0.8,
            config=TransportConfig(n_photons=1500, seed=6),
            keep_directions=True,
        )
        norms = np.linalg.norm(result.final_directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_a_beam_straight_down_does_not_produce_nans(self, backend):
        # u_z = 1 exactly is where the scattering rotation divides by
        # sqrt(1 - u_z^2). The guarded branch must be taken, and the guard
        # must also keep the *unselected* branch from producing a nan that
        # xp.where then propagates.
        result = run_transport_3d(
            backend,
            100.0,
            0.9,
            0.7,
            config=TransportConfig(n_photons=2000, seed=7),
            incidence="collimated",
            keep_directions=True,
        )
        assert np.isfinite(result.final_directions).all()
        assert np.isfinite(result.reflected)

    def test_runs_are_reproducible_from_the_seed(self, backend):
        kwargs = dict(config=TransportConfig(n_photons=1200, seed=11))
        a = run_transport_3d(backend, 100.0, 0.9, 0.4, **kwargs)
        b = run_transport_3d(backend, 100.0, 0.9, 0.4, **kwargs)
        assert a.reflected == b.reflected
        np.testing.assert_array_equal(a.binned_weight, b.binned_weight)

    def test_different_seeds_give_different_realisations(self, backend):
        a = run_transport_3d(
            backend, 100.0, 0.9, 0.4, config=TransportConfig(n_photons=1200, seed=1)
        )
        b = run_transport_3d(
            backend, 100.0, 0.9, 0.4, config=TransportConfig(n_photons=1200, seed=2)
        )
        assert a.reflected != b.reflected


class TestAgainstTheOneDimensionalEngine:
    """A matched boundary makes the two engines the same problem."""

    @pytest.mark.parametrize(
        "omega,g,max_scatters",
        [(0.9, 0.0, 3_000), (0.9, 0.85, 3_000), (0.99, 0.5, 20_000), (0.7, 0.3, 2_000)],
    )
    def test_total_reflectance_agrees(self, backend, omega, g, max_scatters):
        # Same illumination on both sides. The engines disagree by a constant
        # 0.06 if one is given a pencil beam and the other Lambertian light,
        # which is not a bug in either -- it is a different problem.
        n_photons = 20000
        config = TransportConfig(
            n_photons=n_photons, seed=21, max_scatters=max_scatters
        )
        flat = run_transport(backend, 100.0, omega, g, config=config)
        cube = run_transport_3d(
            backend,
            100.0,
            omega,
            g,
            config=config,
            surface_index=1.0,
            incidence="diffuse",
        )
        # Two independent estimators of the same number, so the tolerance is
        # set by the photon count rather than by taste: the combined standard
        # error at 20000 photons is about 0.5% absolute at worst, and this is
        # three of them.
        combined_sigma = np.sqrt(2.0 * flat.albedo * (1.0 - flat.albedo) / n_photons)
        assert abs(cube.reflected - flat.albedo) < 3.0 * combined_sigma + 1e-3

    def test_a_conservative_medium_returns_every_photon(self, backend):
        # No absorption, so this closes at any step budget: whatever has not
        # escaped is still in flight and counted as truncated. A short budget
        # tests the ledger just as well as a long one and costs far less.
        result = run_transport_3d(
            backend,
            100.0,
            1.0,
            0.5,
            config=TransportConfig(n_photons=3000, seed=22, max_scatters=2_000),
            surface_index=1.0,
        )
        assert result.reflected + result.truncated == pytest.approx(1.0, abs=1e-9)

    def test_a_purely_absorbing_medium_reflects_nothing(self, backend):
        result = run_transport_3d(
            backend, 100.0, 0.0, 0.0, config=TransportConfig(n_photons=2000, seed=23)
        )
        assert result.reflected == pytest.approx(0.0, abs=1e-12)


class TestTheInterface:
    def test_fresnel_lowers_the_reflectance_of_an_absorbing_pack(self, backend):
        # Photons turned back at the boundary travel further before they
        # escape, so they absorb more. The direction of the effect is
        # predictable even though its size is not.
        config = TransportConfig(n_photons=20000, seed=31, max_scatters=5_000)
        matched = run_transport_3d(
            backend, 100.0, 0.9, 0.5, config=config, surface_index=1.0
        )
        real = run_transport_3d(
            backend, 100.0, 0.9, 0.5, config=config, surface_index=1.31
        )
        assert real.reflected < matched.reflected

    def test_a_conservative_pack_escapes_whatever_the_interface(self, backend):
        # With no absorption the interface can delay a photon but cannot keep
        # it: total internal reflection is not absorption.
        result = run_transport_3d(
            backend,
            100.0,
            1.0,
            0.5,
            config=TransportConfig(n_photons=2000, seed=32, max_scatters=3_000),
            surface_index=1.31,
        )
        assert result.reflected + result.truncated == pytest.approx(1.0, abs=1e-9)

    def test_rejects_an_index_below_one(self, backend):
        with pytest.raises(ValueError, match="refractive index"):
            run_transport_3d(backend, 100.0, 0.9, 0.0, surface_index=0.5)


class TestRadialBinning:
    def test_the_profile_and_the_overflow_account_for_every_escape(self, backend):
        # The ledger that matters. Whatever left through the surface is
        # either in a bin or counted as outside one; there is no third place
        # for it to go, and a profile that quietly lost weight would still
        # look like a perfectly reasonable curve.
        result = run_transport_3d(
            backend,
            100.0,
            0.95,
            0.5,
            config=TransportConfig(n_photons=8000, seed=41, max_scatters=3_000),
            radial_edges_m=log_radial_edges(1e-4, 10.0, 40),
            incidence="collimated",
        )
        assert result.binned_weight.sum() + result.outside_bins == pytest.approx(
            result.reflected, rel=1e-9
        )

    def test_generous_bins_capture_essentially_all_of_it(self, backend):
        # A pencil beam puts some escapes arbitrarily close to the source, so
        # "wide enough to hold everything" is an aspiration rather than a
        # fact -- the inner edge can always be undercut. Four decades below
        # the mean free path leaves less than a thousandth outside.
        result = run_transport_3d(
            backend,
            100.0,
            0.95,
            0.5,
            config=TransportConfig(n_photons=8000, seed=41, max_scatters=3_000),
            radial_edges_m=log_radial_edges(1e-6, 100.0, 60),
            incidence="collimated",
        )
        assert result.outside_bins < 1e-3 * result.reflected

    def test_weight_outside_the_bins_is_reported_not_dropped(self, backend):
        result = run_transport_3d(
            backend,
            100.0,
            0.95,
            0.5,
            config=TransportConfig(n_photons=4000, seed=42, max_scatters=3_000),
            radial_edges_m=log_radial_edges(1e-3, 2e-3, 4),
            incidence="collimated",
        )
        assert result.outside_bins > 0
        assert result.binned_weight.sum() + result.outside_bins == pytest.approx(
            result.reflected, rel=1e-9
        )

    def test_reflectance_per_area_falls_with_distance(self, backend):
        result = run_transport_3d(
            backend,
            100.0,
            0.98,
            0.5,
            config=TransportConfig(n_photons=40000, seed=43, max_scatters=6_000),
            radial_edges_m=log_radial_edges(2e-3, 0.5, 12),
            incidence="collimated",
        )
        r = result.reflectance
        populated = result.binned_weight > 0
        assert np.all(np.diff(r[populated]) < 0)

    def test_the_annulus_areas_are_right(self, backend):
        e = log_radial_edges(1e-3, 1.0, 5)
        result = run_transport_3d(
            backend, 100.0, 0.9, 0.0, config=TransportConfig(n_photons=500, seed=44),
            radial_edges_m=e,
        )
        np.testing.assert_allclose(
            result.annulus_area_m2, np.pi * (e[1:] ** 2 - e[:-1] ** 2)
        )

    def test_a_collimated_beam_enters_at_the_origin(self, backend):
        # Everything escaping close to the source is what makes a
        # source-detector separation meaningful at all.
        result = run_transport_3d(
            backend,
            1000.0,
            0.9,
            0.0,
            config=TransportConfig(n_photons=4000, seed=45, max_scatters=3_000),
            radial_edges_m=log_radial_edges(1e-5, 1.0, 30),
            incidence="collimated",
        )
        centres = result.bin_centres_m
        weighted_mean = (result.binned_weight * centres).sum() / result.binned_weight.sum()
        assert weighted_mean < 0.05


class TestRadialEdges:
    def test_spans_the_requested_range(self):
        e = log_radial_edges(1e-3, 1.0, 10)
        assert e[0] == pytest.approx(1e-3)
        assert e[-1] == pytest.approx(1.0)
        assert e.size == 11

    def test_is_logarithmic(self):
        e = log_radial_edges(1e-3, 1.0, 10)
        ratios = e[1:] / e[:-1]
        np.testing.assert_allclose(ratios, ratios[0])

    def test_rejects_a_non_positive_inner_edge(self):
        with pytest.raises(ValueError):
            log_radial_edges(0.0, 1.0, 10)


class TestValidation:
    def test_rejects_an_out_of_range_albedo(self, backend):
        with pytest.raises(ValueError):
            run_transport_3d(backend, 100.0, 1.5, 0.0)

    def test_rejects_a_nonpositive_extinction_coefficient(self, backend):
        with pytest.raises(ValueError):
            run_transport_3d(backend, 0.0, 0.9, 0.0)

    def test_rejects_an_unknown_incidence(self, backend):
        with pytest.raises(ValueError, match="incidence"):
            run_transport_3d(backend, 100.0, 0.9, 0.0, incidence="sideways")
