"""An object in the snowpack, and what must remain true once there is one.

The hard part of adding a second medium is not the optics, it is proving that
nothing about the first one changed. A region bug does not raise: photons
simply carry the wrong extinction for part of their path, the profile stays
smooth, and the ledger still balances. So the anchor test here is an object
made of snow -- same extinction, same albedo, same asymmetry, same index --
which must be invisible.

The physics tests then ask only for directions that can be argued in advance:
an absorber removes light, a void does not, and an object below the depth the
light reaches cannot matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.domain.geometry import Box
from snow_mcrt.domain.transport import TransportConfig
from snow_mcrt.domain.transport3d import (
    BuriedObject,
    log_radial_edges,
    run_transport_3d,
)

# A deliberately absorbing test medium: the penetration depth works out at
# 1.45 cm, so an object has to sit within a couple of centimetres to be lit at
# all. Burying it at ten -- seven penetration depths, where one part in a
# thousand of the light arrives -- makes every object invisible and every test
# vacuous. That is a property of the snowpack, not of the engine.
OMEGA, G, BETA = 0.99, 0.85, 1000.0
SHALLOW = Box(lower=np.array([-0.1, -0.1, 0.005]), upper=np.array([0.1, 0.1, 0.205]))


@pytest.fixture
def backend():
    return NumpyBackend()


def run(backend, obj=None, seed=3, photons=30000, steps=3000):
    return run_transport_3d(
        backend,
        BETA,
        OMEGA,
        G,
        config=TransportConfig(n_photons=photons, seed=seed, max_scatters=steps),
        incidence="collimated",
        surface_index=1.31,
        radial_edges_m=log_radial_edges(2e-3, 1.0, 14),
        obj=obj,
    )


class TestAnObjectMadeOfSnowIsInvisible:
    """The anchor. If this drifts, every other number here is unmoored."""

    def test_reflectance_is_unchanged(self, backend):
        made_of_snow = BuriedObject(
            SHALLOW,
            extinction_coefficient=BETA,
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            refractive_index=1.31,
        )
        without = run(backend)
        with_it = run(backend, made_of_snow)
        # Not bit-identical: crossing a face splits a step in two and a fresh
        # free path is drawn on the far side. That is the memoryless property
        # doing its job, so the realisations differ and the ensembles must
        # not.
        assert with_it.reflected == pytest.approx(without.reflected, abs=0.01)

    def test_the_profile_is_unchanged(self, backend):
        made_of_snow = BuriedObject(
            SHALLOW,
            extinction_coefficient=BETA,
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            refractive_index=1.31,
        )
        photons = 30000
        without = run(backend, photons=photons)
        with_it = run(backend, made_of_snow, photons=photons)

        # Compare bin by bin against that bin's own noise, not against a flat
        # tolerance. The profile spans orders of magnitude, so a fixed
        # percentage is either vacuous in the near field or guaranteed to
        # fail in the tail -- where a bin may hold a handful of photons and
        # its one-sigma spread is tens of percent.
        counts = without.binned_weight * photons
        resolved = counts > 100
        assert resolved.sum() >= 5
        sigma = 1.0 / np.sqrt(counts[resolved])
        deviation = (
            np.abs(with_it.binned_weight[resolved] - without.binned_weight[resolved])
            / without.binned_weight[resolved]
        )
        assert np.all(deviation < 4.0 * sigma)

    def test_the_ledger_still_balances(self, backend):
        made_of_snow = BuriedObject(
            SHALLOW,
            extinction_coefficient=BETA,
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            refractive_index=1.31,
        )
        result = run(backend, made_of_snow)
        assert abs(result.energy_balance) < 100 * result.config.roulette_threshold


class TestAnAbsorberRemovesLight:
    def test_a_black_slab_darkens_the_return(self, backend):
        black = BuriedObject(
            SHALLOW,
            extinction_coefficient=1e5,
            single_scattering_albedo=0.0,
            refractive_index=1.31,
        )
        assert run(backend, black).reflected < run(backend).reflected

    def test_burying_it_deeper_matters_less(self, backend):
        # The whole premise of the detectability work: sensitivity falls with
        # depth. If this came out flat, the engine would not be measuring
        # depth at all.
        shallow = BuriedObject(
            SHALLOW, extinction_coefficient=1e5, refractive_index=1.31
        )
        deep = BuriedObject(
            Box(lower=np.array([-0.1, -0.1, 0.05]), upper=np.array([0.1, 0.1, 0.25])),
            extinction_coefficient=1e5,
            refractive_index=1.31,
        )
        plain = run(backend).reflected
        assert abs(run(backend, deep).reflected - plain) < abs(
            run(backend, shallow).reflected - plain
        )

    def test_a_wider_slab_removes_more(self, backend):
        narrow = BuriedObject(
            Box(np.array([-0.02, -0.02, 0.005]), np.array([0.02, 0.02, 0.205])),
            extinction_coefficient=1e5,
            refractive_index=1.31,
        )
        wide = BuriedObject(
            Box(np.array([-0.3, -0.3, 0.005]), np.array([0.3, 0.3, 0.205])),
            extinction_coefficient=1e5,
            refractive_index=1.31,
        )
        assert run(backend, wide).reflected < run(backend, narrow).reflected


class TestAVoid:
    """A cavity is the interesting case, and the one that breaks arithmetic.

    ``beta = 0`` means no collision length at all. Sampling a free path as
    ``-log(u)/beta`` there is a division by zero, and the photon's position
    becomes a nan that spreads through every subsequent comparison without
    raising.
    """

    def test_does_not_produce_nans(self, backend):
        void = BuriedObject(SHALLOW, extinction_coefficient=0.0, refractive_index=1.0)
        result = run(backend, void)
        assert np.isfinite(result.reflected)
        assert np.isfinite(result.binned_weight).all()
        assert abs(result.energy_balance) < 100 * result.config.roulette_threshold

    def test_darkens_the_pack_despite_absorbing_nothing(self, backend):
        """A cavity absorbs nothing and still takes light away.

        The direction surprises people, so it is worth stating why. Inside a
        void there is nothing to scatter off, so a photon that gets in
        travels in a straight line to the far wall -- twenty centimetres
        down, more than ten penetration depths, where it is lost to the snow
        below. The cavity is a light pipe pointing away from the detector.

        This is the effect ``docs/detectability.md`` predicted would make
        voids easier to find than absorbers of the same size, and here it is
        measured: the void removes 0.14 of a 0.35 return.
        """
        void = BuriedObject(SHALLOW, extinction_coefficient=0.0, refractive_index=1.0)
        with_void = run(backend, void)
        plain = run(backend)
        assert with_void.reflected < plain.reflected - 0.05
        # Energy is conserved, so what stops coming back is deposited in the
        # snow instead.
        assert with_void.absorbed > plain.absorbed


class TestTheInterfaceIsReal:
    def test_an_index_step_alone_changes_the_answer(self, backend):
        # Identical scattering and absorption, only the index differs. Any
        # difference is the Fresnel faces and nothing else.
        optical_twin = BuriedObject(
            SHALLOW,
            extinction_coefficient=BETA,
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            refractive_index=1.8,
        )
        matched = BuriedObject(
            SHALLOW,
            extinction_coefficient=BETA,
            single_scattering_albedo=OMEGA,
            asymmetry=G,
            refractive_index=1.31,
        )
        assert run(backend, optical_twin).reflected != pytest.approx(
            run(backend, matched).reflected, abs=2e-3
        )


class TestValidation:
    def test_rejects_a_negative_extinction(self):
        with pytest.raises(ValueError, match="non-negative"):
            BuriedObject(SHALLOW, extinction_coefficient=-1.0)

    def test_rejects_an_albedo_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match="albedo"):
            BuriedObject(SHALLOW, single_scattering_albedo=1.5)

    def test_rejects_an_index_below_one(self):
        with pytest.raises(ValueError, match="refractive index"):
            BuriedObject(SHALLOW, refractive_index=0.9)
