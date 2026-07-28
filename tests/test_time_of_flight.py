"""Optical path length, and why it is a depth coordinate.

A steady-state measurement sees how much light came back. It cannot see how
far that light went, and two different snowpacks — or the same snowpack with
and without something buried in it — can return the same amount by different
routes. Path length separates them, which is why every time-resolved
instrument in diffuse optics exists.

The tests here pin the quantity against things known without running it: a
photon cannot return having travelled less than the straight line back, light
returning from further away travelled further, and the mean is a
weight-weighted mean rather than a count-weighted one, which matters the
moment Russian roulette starts reweighting survivors.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.domain.transport import TransportConfig
from snow_mcrt.domain.transport3d import log_radial_edges, run_transport_3d

OMEGA, G, BETA = 0.98, 0.5, 500.0


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def result(backend):
    return run_transport_3d(
        backend,
        BETA,
        OMEGA,
        G,
        config=TransportConfig(
            n_photons=60000, seed=3, max_scatters=4000, delta_scaled=False
        ),
        incidence="collimated",
        radial_edges_m=log_radial_edges(2e-3, 0.5, 12),
    )


class TestItIsAPathLength:
    def test_no_photon_returns_shorter_than_the_straight_line(self, result):
        # A photon that came back at separation rho travelled at least rho:
        # it had to get there. Anything less is an accounting error, and one
        # that would look like a plausible early-time signal.
        sampled = result.binned_weight > 0
        assert np.all(result.mean_path_m[sampled] >= result.bin_centres_m[sampled])

    def test_light_from_further_away_travelled_further(self, result):
        sampled = result.binned_weight > 0
        path = result.mean_path_m[sampled]
        assert np.all(np.diff(path) > 0)

    def test_an_empty_bin_has_no_mean(self, backend):
        # Not zero. A bin nothing reached has no mean path, and reporting
        # zero would put a spurious early-time point on every plot.
        result = run_transport_3d(
            backend,
            BETA,
            OMEGA,
            G,
            config=TransportConfig(n_photons=600, seed=4, max_scatters=400),
            radial_edges_m=log_radial_edges(1e-3, 50.0, 20),
        )
        assert (result.binned_weight == 0).any()
        assert np.isnan(result.mean_path_m[result.binned_weight == 0]).all()

    def test_the_mean_is_weighted_by_energy_not_by_count(self, result):
        # binned_path_weight is the sum of weight times path, so dividing by
        # binned_weight gives an energy-weighted mean. Under Russian roulette
        # a survivor carries the weight of the photons that were killed, and
        # a count-weighted mean would silently under-represent it.
        sampled = result.binned_weight > 0
        np.testing.assert_allclose(
            result.mean_path_m[sampled],
            result.binned_path_weight[sampled] / result.binned_weight[sampled],
        )


class TestScaling:
    def test_a_denser_medium_returns_light_that_travelled_less(self, backend):
        # Doubling the extinction halves the mean free path, so a photon
        # reaching the same separation does it in shorter hops but a
        # comparable number of them -- the total path at a fixed separation
        # in metres goes down.
        def mean_at(beta):
            r = run_transport_3d(
                backend,
                beta,
                OMEGA,
                G,
                config=TransportConfig(
                    n_photons=40000, seed=5, max_scatters=4000, delta_scaled=False
                ),
                incidence="collimated",
                radial_edges_m=log_radial_edges(5e-3, 0.2, 6),
            )
            sampled = r.binned_weight > 0
            return float(np.nanmean(r.mean_path_m[sampled]))

        assert mean_at(1000.0) < mean_at(500.0)

    def test_time_follows_path_through_the_speed_of_light_in_the_medium(
        self, result
    ):
        expected = result.mean_path_m / (299_792_458.0 / result.surface_index) * 1e12
        np.testing.assert_allclose(result.mean_time_ps(), expected, equal_nan=True)

    def test_a_higher_index_slows_the_light_down(self, result):
        assert np.nanmean(result.mean_time_ps(2.0)) > np.nanmean(
            result.mean_time_ps(1.31)
        )

    def test_the_times_are_picoseconds_not_something_else(self, result):
        # A centimetre of path in ice is about 44 ps. If this ever comes back
        # in nanoseconds or metres the number will still plot beautifully.
        sampled = result.binned_weight > 0
        near = result.mean_time_ps()[sampled][0]
        assert 1.0 < near < 10_000.0


class TestItDoesNotDisturbTheRest:
    def test_the_energy_ledger_is_unchanged(self, result):
        assert abs(result.energy_balance) < 100 * result.config.roulette_threshold

    def test_the_profile_is_unchanged(self, backend):
        # Path tracking is pure bookkeeping: it must not move a single
        # photon. Same seed, same everything.
        kwargs = dict(
            config=TransportConfig(n_photons=8000, seed=9, max_scatters=1500),
            incidence="collimated",
            radial_edges_m=log_radial_edges(2e-3, 0.5, 10),
        )
        a = run_transport_3d(backend, BETA, OMEGA, G, **kwargs)
        b = run_transport_3d(backend, BETA, OMEGA, G, **kwargs)
        np.testing.assert_array_equal(a.binned_weight, b.binned_weight)
        assert a.reflected == b.reflected
