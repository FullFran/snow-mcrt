"""Transport validated against exact invariants and an accurate oracle.

Ordered by how much each test constrains. The exact invariants come first --
a conservative slab must return every photon, and the energy ledger must
balance to floating point -- because they hold for any parameters and admit no
tuning. The oracle comparisons follow.

Note which oracle. ``semi_infinite_albedo`` (two-stream) is 8.9% high at
``omega = 0.9``, so it cannot judge a transport run outside the conservative
limit. ``van_de_hulst_semi_infinite_albedo`` is good to ~1% throughout and is
what the quantitative tests use.
"""

import numpy as np
import pytest

from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.domain.analytic import (
    similarity_scaled_albedo,
    van_de_hulst_semi_infinite_albedo,
)
from snow_mcrt.domain.transport import TransportConfig, run_transport

BETA = 5000.0  # m^-1, a 0.2 mm mean free path -- seasonal snow


@pytest.fixture
def backend():
    return NumpyBackend()


def config(**kwargs) -> TransportConfig:
    base = dict(n_photons=20_000, seed=1, max_scatters=8_000)
    base.update(kwargs)
    return TransportConfig(**base)


class TestExactInvariants:
    """Hold for any parameters, and admit no tuning."""

    @pytest.mark.parametrize("g", [0.0, 0.85])
    def test_a_conservative_slab_returns_every_photon(self, backend, g):
        # No absorption means no sink: everything launched must come out one
        # side or the other. Exact to floating point, not statistically.
        result = run_transport(
            backend, BETA, 1.0, g, thickness_m=0.002, config=config(max_scatters=50_000)
        )
        assert result.truncated == 0.0
        assert result.absorbed == 0.0
        assert result.reflected + result.transmitted == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("omega,g", [(0.5, 0.0), (0.9, 0.85), (0.999, 0.89)])
    def test_the_energy_ledger_balances_exactly_without_roulette(
        self, backend, omega, g
    ):
        # With roulette disabled nothing is ever discarded, so the ledger is
        # an identity rather than an estimate: it closes to float64 epsilon.
        result = run_transport(
            backend, BETA, omega, g, config=config(roulette_threshold=0.0)
        )
        assert result.energy_balance == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("omega,g", [(0.5, 0.0), (0.9, 0.85), (0.999, 0.89)])
    def test_the_energy_ledger_balances_in_expectation_with_roulette(
        self, backend, omega, g
    ):
        """Roulette conserves energy in expectation, not per realisation.

        Killed photons take their weight with them; survivors are boosted by
        1/p to compensate. The two cancel on average, so the ledger closes to
        within the sampling noise -- of order the roulette threshold -- and
        the residual changes sign between runs. Demanding exactness here
        would be demanding that an unbiased estimator be a conserved
        quantity, which it is not.
        """
        result = run_transport(backend, BETA, omega, g, config=config())
        residual = abs(result.energy_balance - 1.0)
        assert 0 < residual < 100 * config().roulette_threshold

    def test_all_channels_are_non_negative(self, backend):
        result = run_transport(backend, BETA, 0.9, 0.5, config=config())
        assert result.reflected >= 0
        assert result.transmitted >= 0
        assert result.absorbed >= 0
        assert result.truncated >= 0

    def test_a_purely_absorbing_medium_reflects_nothing(self, backend):
        result = run_transport(backend, BETA, 0.0, 0.0, config=config())
        assert result.reflected == 0.0
        assert result.absorbed == pytest.approx(1.0, abs=1e-12)

    def test_runs_are_reproducible_from_the_seed(self, backend):
        first = run_transport(backend, BETA, 0.95, 0.5, config=config(seed=99))
        second = run_transport(backend, BETA, 0.95, 0.5, config=config(seed=99))
        assert first.reflected == second.reflected
        assert first.absorbed == second.absorbed

    def test_different_seeds_give_different_realisations(self, backend):
        first = run_transport(backend, BETA, 0.95, 0.5, config=config(seed=1))
        second = run_transport(backend, BETA, 0.95, 0.5, config=config(seed=2))
        assert first.reflected != second.reflected


class TestAgainstTheOracle:
    @pytest.mark.parametrize(
        "omega,max_scatters", [(0.5, 2_000), (0.8, 4_000), (0.9, 8_000), (0.95, 15_000)]
    )
    def test_isotropic_albedo_matches_van_de_hulst(
        self, backend, omega, max_scatters
    ):
        result = run_transport(
            backend,
            BETA,
            omega,
            0.0,
            config=config(n_photons=40_000, max_scatters=max_scatters),
        )
        expected = float(van_de_hulst_semi_infinite_albedo(omega))
        # The reference is itself a ~1% fit, so this is the floor.
        assert result.albedo == pytest.approx(expected, rel=0.025)

    def test_accuracy_improves_towards_the_conservative_limit(self, backend):
        # Where snow actually lives. The fit is tightest here and so is the
        # agreement.
        result = run_transport(
            backend, BETA, 0.99, 0.0, config=config(n_photons=40_000, max_scatters=40_000)
        )
        expected = float(van_de_hulst_semi_infinite_albedo(0.99))
        assert result.albedo == pytest.approx(expected, rel=0.005)

    @pytest.mark.parametrize("omega,g", [(0.99, 0.85), (0.999, 0.89)])
    def test_anisotropic_albedo_matches_similarity_scaling(self, backend, omega, g):
        # Exercises the angular sampling. If the deflection rotation were
        # wrong, forward scattering would not map onto its isotropic
        # equivalent at all.
        result = run_transport(
            backend,
            BETA,
            omega,
            g,
            config=config(n_photons=40_000, max_scatters=60_000, delta_scaled=False),
        )
        expected = float(similarity_scaled_albedo(omega, g))
        assert result.albedo == pytest.approx(expected, rel=0.02)


class TestGeometry:
    def test_an_optically_thick_slab_behaves_as_semi_infinite(self, backend):
        thick = run_transport(
            backend, BETA, 0.9, 0.0, thickness_m=0.2, config=config()
        )
        infinite = run_transport(backend, BETA, 0.9, 0.0, config=config())
        assert thick.albedo == pytest.approx(infinite.albedo, abs=1e-9)
        assert thick.transmitted == 0.0

    def test_a_thinner_slab_transmits_more(self, backend):
        thin = run_transport(
            backend, BETA, 0.95, 0.0, thickness_m=0.0005, config=config()
        )
        thick = run_transport(
            backend, BETA, 0.95, 0.0, thickness_m=0.005, config=config()
        )
        assert thin.transmitted > thick.transmitted
        assert thin.albedo < thick.albedo

    def test_grazing_incidence_reflects_more_than_normal(self, backend):
        # A shallow entry angle keeps photons near the surface, so fewer
        # reach depth before turning around.
        normal = run_transport(backend, BETA, 0.9, 0.0, config=config(), incidence=1.0)
        grazing = run_transport(backend, BETA, 0.9, 0.0, config=config(), incidence=0.2)
        assert grazing.albedo > normal.albedo

    def test_rejects_grazing_incidence_that_never_enters(self, backend):
        with pytest.raises(ValueError, match="grazing"):
            run_transport(backend, BETA, 0.9, 0.0, config=config(), incidence=0.0)


class TestConvergence:
    def test_albedo_converges_as_photons_are_added(self, backend):
        reference = run_transport(
            backend, BETA, 0.9, 0.0, config=config(n_photons=200_000, seed=7)
        )
        coarse = run_transport(
            backend, BETA, 0.9, 0.0, config=config(n_photons=2_000, seed=7)
        )
        fine = run_transport(
            backend, BETA, 0.9, 0.0, config=config(n_photons=50_000, seed=7)
        )
        assert abs(fine.albedo - reference.albedo) < abs(
            coarse.albedo - reference.albedo
        )

    def test_truncated_weight_is_reported_not_hidden(self, backend):
        """A convergence failure must be visible, not folded into absorption.

        Cut the scattering budget to almost nothing. The run still balances,
        but the shortfall shows up in `truncated` where a reader can see it.
        Attributing it to absorption instead would produce a plausible albedo
        and no indication anything was wrong.
        """
        starved = run_transport(
            backend, BETA, 0.999, 0.0, config=config(max_scatters=5)
        )
        assert starved.scatters == 5
        assert starved.truncated > 0.5
        assert starved.energy_balance == pytest.approx(1.0, abs=1e-9)

    def test_a_generous_budget_leaves_nothing_in_flight(self, backend):
        result = run_transport(
            backend, BETA, 0.9, 0.0, config=config(max_scatters=20_000)
        )
        assert result.truncated < 1e-6


class TestRussianRoulette:
    def test_does_not_bias_the_albedo(self, backend):
        # Survivors are boosted by 1/p, so killing faint photons must not
        # change the expectation. If the boost were dropped, roulette would
        # quietly eat energy and the albedo would fall.
        with_roulette = run_transport(
            backend, BETA, 0.9, 0.0, config=config(n_photons=100_000, seed=5)
        )
        without = run_transport(
            backend,
            BETA,
            0.9,
            0.0,
            config=config(n_photons=100_000, seed=5, roulette_threshold=0.0),
        )
        assert with_roulette.albedo == pytest.approx(without.albedo, rel=0.02)

    def test_shortens_the_run(self, backend):
        with_roulette = run_transport(
            backend, BETA, 0.5, 0.0, config=config(max_scatters=5_000)
        )
        without = run_transport(
            backend, BETA, 0.5, 0.0, config=config(max_scatters=5_000, roulette_threshold=0.0)
        )
        assert with_roulette.scatters < without.scatters


class TestValidation:
    def test_rejects_an_out_of_range_albedo(self, backend):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run_transport(backend, BETA, 1.5, 0.0, config=config())

    def test_rejects_a_nonpositive_extinction_coefficient(self, backend):
        with pytest.raises(ValueError, match="must be positive"):
            run_transport(backend, 0.0, 0.9, 0.0, config=config())

    def test_rejects_a_zero_photon_run(self):
        with pytest.raises(ValueError, match="at least one photon"):
            TransportConfig(n_photons=0)

    def test_rejects_an_impossible_roulette_probability(self):
        with pytest.raises(ValueError, match="roulette survival"):
            TransportConfig(roulette_survival=1.0)
