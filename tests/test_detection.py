"""Contrast against depth, and the noise floor that decides what it means.

The trap this module exists to avoid is a plot. Contrast falls by orders of
magnitude with depth, so it is drawn on a log axis, and on a log axis pure
Monte Carlo noise draws a perfectly convincing curve continuing to the edge of
the figure. Every test here is really about telling a measurement from its
noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.application.detection import sweep_burial_depth
from snow_mcrt.domain.transport import TransportConfig

OMEGA, G, BETA = 0.99, 0.85, 1000.0


@pytest.fixture(scope="module")
def backend():
    return NumpyBackend()


# Module-scoped. This is a full transport run and every test in the file
# reads it; at function scope pytest recomputes it once per test.
@pytest.fixture(scope="module")
def sweep(backend):
    return sweep_burial_depth(
        backend,
        OMEGA,
        G,
        BETA,
        depths_m=np.array([0.002, 0.01, 0.05]),
        label="black slab",
        config=TransportConfig(n_photons=12000, seed=4, max_scatters=2000),
    )


class TestTheCurve:
    def test_contrast_is_negative_for_an_absorber(self, sweep):
        assert sweep.contrast[0] < 0

    def test_contrast_fades_with_depth(self, sweep):
        assert abs(sweep.contrast[0]) > abs(sweep.contrast[1]) > abs(
            sweep.contrast[2]
        )

    def test_a_void_also_removes_light(self, backend):
        # And it absorbs nothing. Inside a cavity there is nothing to scatter
        # off, so a photon that enters runs straight to the far wall well
        # below the penetration depth and is lost downward.
        void = sweep_burial_depth(
            backend,
            OMEGA,
            G,
            BETA,
            depths_m=np.array([0.002]),
            label="void",
            object_extinction=0.0,
            object_index=1.0,
            config=TransportConfig(n_photons=12000, seed=4, max_scatters=2000),
        )
        assert void.contrast[0] < -0.1

    def test_the_reference_is_shared_across_depths(self, sweep):
        assert np.isscalar(sweep.plain_reflected) or np.ndim(
            sweep.plain_reflected
        ) == 0
        np.testing.assert_allclose(
            sweep.contrast,
            (sweep.reflected - sweep.plain_reflected) / sweep.plain_reflected,
        )


class TestTheNoiseFloor:
    def test_is_positive_and_falls_with_photon_count(self, backend):
        kwargs = dict(
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            extinction_coefficient=BETA,
            depths_m=np.array([0.01]),
            label="black slab",
        )
        few = sweep_burial_depth(
            backend, config=TransportConfig(n_photons=3000, seed=1), **kwargs
        )
        many = sweep_burial_depth(
            backend, config=TransportConfig(n_photons=12000, seed=1), **kwargs
        )
        assert 0 < many.noise_floor < few.noise_floor

    def test_scales_as_one_over_root_n(self, backend):
        kwargs = dict(
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            extinction_coefficient=BETA,
            depths_m=np.array([0.01]),
            label="black slab",
        )
        few = sweep_burial_depth(
            backend, config=TransportConfig(n_photons=3000, seed=1), **kwargs
        )
        many = sweep_burial_depth(
            backend, config=TransportConfig(n_photons=12000, seed=1), **kwargs
        )
        assert few.noise_floor / many.noise_floor == pytest.approx(2.0, rel=0.05)

    def test_a_shallow_object_clears_it_and_a_deep_one_does_not(self, sweep):
        # The whole point of carrying the floor around. Without it, the last
        # point on a log plot is indistinguishable from a detection.
        assert sweep.detectable[0]
        assert not sweep.detectable[-1]


class TestTheTransferableCoordinate:
    def test_depth_is_reported_in_penetration_depths(self, sweep):
        np.testing.assert_allclose(
            sweep.depth_in_penetration_depths,
            sweep.depth_m / sweep.penetration_depth_m,
        )

    def test_the_penetration_depth_matches_the_diffusion_parameters(self, sweep):
        from snow_mcrt.domain.diffusion import DiffusionParameters

        expected = DiffusionParameters.from_optical_properties(
            OMEGA, G, BETA, refractive_index=1.31
        )
        assert sweep.penetration_depth_m == pytest.approx(expected.penetration_depth)


class TestOutput:
    def test_columns_are_all_the_same_length(self, sweep):
        lengths = {len(v) for v in sweep.columns().values()}
        assert lengths == {sweep.depth_m.size}

    def test_carries_what_produced_it(self, sweep):
        assert sweep.label == "black slab"
        assert sweep.n_photons == 12000
