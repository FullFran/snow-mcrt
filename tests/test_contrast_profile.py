"""The measurement, rather than the shadow of it.

Integrating over separation was the wrong thing to report and these tests pin
why. Light returning close to the source never went deep enough to meet
anything buried; light returning far away had to travel through the depth the
object occupies. Separation is a depth selector, so a total return dilutes the
signal with the part of the field that cannot possibly carry it.

The other thing pinned here is that a spectacular contrast is not a detection.
Contrast climbs towards 100% in the far bins precisely because the object
blocks everything, and those bins are also where almost nothing arrives. Only
the product of signal and photon count decides anything, which is why ``best``
maximises signal-to-noise rather than contrast.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.application.detection import measure_contrast_profile
from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.transport import TransportConfig

# omega = 0.999, not 0.99. The ratio that decides whether separation can
# select depth at all is delta / mfp', the number of transport lengths light
# penetrates: at omega = 0.99 it is 2.3, so a shallow object blocks the whole
# field and there is no near field to be insensitive. Real snow is nearer 22.
# This sits at 7 -- enough structure to test against, cheap enough to run.
OMEGA, G, BETA = 0.999, 0.85, 1000.0


@pytest.fixture(scope="module")
def backend():
    return NumpyBackend()


# The penetration depth of THIS medium, taken from the medium rather than
# remembered from another one. Quoting a depth in centimetres and applying it
# to a different snowpack is the mistake this whole module exists to make
# impossible, and it is just as easy to make in a test.
DELTA = DiffusionParameters.from_optical_properties(
    OMEGA, G, BETA, refractive_index=1.31
).penetration_depth


# Module-scoped: this costs two full transport runs and eight tests read it.
# At function scope pytest recomputed it eight times and turned a unit-test
# file into a twenty-minute one. Nothing here mutates the result.
@pytest.fixture(scope="module")
def profile(backend):
    return measure_contrast_profile(
        backend,
        OMEGA,
        G,
        BETA,
        depth_m=0.5 * DELTA,
        config=TransportConfig(n_photons=40000, seed=5, max_scatters=8000),
        n_bins=12,
    )


class TestSeparationSelectsDepth:
    def test_the_closest_separation_is_the_least_sensitive(self, profile):
        # Photons that come back closest to the source went shallowest, so
        # they carry the least about anything buried. The innermost bin here
        # is still half a penetration depth out -- one transport mean free
        # path is already 0.46 delta in this medium, and inside that the
        # profile is single-scattered light rather than a measurement.
        ok = profile.sampled & np.isfinite(profile.contrast)
        assert abs(profile.contrast[ok][0]) < 0.2 * abs(
            profile.contrast[ok][-1]
        )

    def test_contrast_grows_with_separation(self, profile):
        ok = profile.sampled & np.isfinite(profile.contrast)
        c = np.abs(profile.contrast[ok])
        # Not strictly monotonic bin by bin at finite photon counts, but the
        # far half must be unambiguously stronger than the near half.
        half = c.size // 2
        assert c[half:].mean() > 3 * c[:half].mean()

    def test_resolving_by_separation_beats_integrating(self, profile):
        # The headline. The total return dilutes the signal across a field
        # most of which never met the object.
        #
        # The size of the gain is set by delta / mfp', the number of transport
        # lengths light penetrates: here that is 7 and the gain is about 3x.
        # In real snow, nearer 22, it measures 7.5x. The threshold is the
        # floor for this medium, not a universal claim.
        total = abs(
            (profile.with_object.sum() - profile.plain.sum()) / profile.plain.sum()
        )
        best = abs(profile.contrast[profile.best])
        assert best > 2 * total


class TestSignalToNoiseDecides:
    def test_the_best_bin_is_not_the_largest_contrast(self, profile):
        ok = profile.sampled & np.isfinite(profile.contrast)
        largest = int(np.nanargmax(np.where(ok, np.abs(profile.contrast), np.nan)))
        assert profile.snr[profile.best] >= profile.snr[largest]

    def test_noise_grows_as_the_bins_empty(self, profile):
        ok = profile.sampled
        sigma = profile.sigma[ok]
        assert sigma[-1] > sigma[len(sigma) // 2]

    def test_an_unreached_bin_has_infinite_noise_not_zero(self, backend):
        # A bin nothing arrived in must not present itself as a perfect
        # measurement of zero.
        profile = measure_contrast_profile(
            backend,
            OMEGA,
            G,
            BETA,
            depth_m=0.03,
            config=TransportConfig(n_photons=1500, seed=6, max_scatters=600),
            outer_depths=60.0,
            n_bins=24,
        )
        empty = profile.plain == 0
        assert empty.any()
        assert np.all(np.isinf(profile.sigma[empty]))
        # No signal and no noise is not a signal-to-noise of zero, it is the
        # absence of a measurement. `best` filters on isfinite for exactly
        # this reason.
        assert np.all(np.isnan(profile.snr[empty]))

    def test_more_photons_lower_the_noise(self, backend):
        def sigma_at(n):
            p = measure_contrast_profile(
                backend,
                OMEGA,
                G,
                BETA,
                depth_m=0.03,
                config=TransportConfig(n_photons=n, seed=7, max_scatters=2000),
                n_bins=8,
            )
            return float(np.nanmedian(p.sigma[p.sampled]))

        assert sigma_at(20000) < sigma_at(5000)


class TestThePathChannel:
    def test_an_absorber_shortens_the_light_that_survives(self, profile):
        # The second, independent observable. An absorber removes the photons
        # that went deepest, so what comes back travelled less far than it
        # would have. The intensity and the path carry different information
        # and a time-resolved instrument reads both.
        ok = profile.sampled & np.isfinite(profile.path_contrast)
        deep = profile.path_contrast[ok & (profile.rho_in_penetration_depths > 0.5)]
        assert deep.size
        assert deep.mean() < 0

    def test_paths_are_at_least_the_separation(self, profile):
        ok = profile.sampled & np.isfinite(profile.mean_path_plain_m)
        assert np.all(profile.mean_path_plain_m[ok] >= profile.rho_m[ok])


class TestOutput:
    def test_columns_are_all_the_same_length(self, profile):
        lengths = {len(v) for v in profile.columns().values()}
        assert lengths == {profile.rho_m.size}

    def test_carries_the_depth_it_was_run_at(self, profile):
        assert profile.depth_m == pytest.approx(0.5 * DELTA)
