"""One reference run, shared across burial depths.

Deliberately tiny. What matters here is not a number but a structural claim:
the snowpack with nothing in it does not depend on where the object would have
been, so every profile in a sweep must be measured against the *same*
reference. Running it again per depth wastes half the sweep and, worse, puts an
independent realisation of the reference under every point — adding noise to a
comparison whose whole job is to be quiet.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.application.detection import (
    measure_contrast_profile,
    sweep_contrast_profiles,
)
from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.transport import TransportConfig

OMEGA, G, BETA = 0.99, 0.5, 1000.0
DELTA = DiffusionParameters.from_optical_properties(
    OMEGA, G, BETA, refractive_index=1.31
).penetration_depth
CHEAP = TransportConfig(n_photons=2000, seed=2, max_scatters=600)


@pytest.fixture(scope="module")
def profiles():
    return sweep_contrast_profiles(
        NumpyBackend(),
        OMEGA,
        G,
        BETA,
        depths_m=np.array([0.3, 0.9]) * DELTA,
        config=CHEAP,
        n_bins=6,
    )


class TestTheReferenceIsShared:
    def test_every_depth_is_measured_against_the_same_snowpack(self, profiles):
        # Bit-identical, not merely close. Two realisations of the reference
        # would agree statistically and still put noise into every contrast.
        first, second = profiles
        np.testing.assert_array_equal(first.plain, second.plain)
        np.testing.assert_array_equal(
            first.mean_path_plain_m, second.mean_path_plain_m
        )

    def test_the_objects_differ(self, profiles):
        first, second = profiles
        assert not np.array_equal(first.with_object, second.with_object)

    def test_one_profile_per_depth_in_order(self, profiles):
        assert len(profiles) == 2
        assert profiles[0].depth_m < profiles[1].depth_m


class TestItAgreesWithTheSingleShotVersion:
    def test_same_answer_as_measuring_one_depth_alone(self):
        # The sweep is an optimisation, not a different calculation. Same
        # seed, same bins, same object, so the same numbers -- otherwise the
        # cheap path and the careful path have quietly diverged.
        depth = 0.3 * DELTA
        alone = measure_contrast_profile(
            NumpyBackend(),
            OMEGA,
            G,
            BETA,
            depth_m=depth,
            config=CHEAP,
            n_bins=6,
            outer_depths=6.0,
        )
        swept = sweep_contrast_profiles(
            NumpyBackend(),
            OMEGA,
            G,
            BETA,
            depths_m=np.array([depth]),
            config=CHEAP,
            n_bins=6,
        )[0]
        np.testing.assert_allclose(alone.plain, swept.plain)
        np.testing.assert_allclose(alone.with_object, swept.with_object)
        np.testing.assert_allclose(alone.rho_m, swept.rho_m)


class TestItStaysCheap:
    def test_a_sweep_of_n_depths_costs_n_plus_one_runs(self, monkeypatch):
        # The point of the function, asserted rather than assumed. Counting
        # calls is the only way to notice if someone reintroduces a
        # per-depth reference.
        import snow_mcrt.application.detection as detection

        calls = []
        original = detection.run_transport_3d

        def counting(*args, **kwargs):
            calls.append(kwargs.get("obj"))
            return original(*args, **kwargs)

        monkeypatch.setattr(detection, "run_transport_3d", counting)
        sweep_contrast_profiles(
            NumpyBackend(),
            OMEGA,
            G,
            BETA,
            depths_m=np.array([0.3, 0.6, 0.9]) * DELTA,
            config=CHEAP,
            n_bins=5,
        )
        assert len(calls) == 4
        assert calls[0] is None
        assert all(obj is not None for obj in calls[1:])
