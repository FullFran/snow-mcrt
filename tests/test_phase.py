"""Phase functions: normalisation, sampling, and the forward-peak trap."""

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.adapters.numpy_backend import NumpyBackend
from snow_mcrt.domain.phase import (
    TabulatedPhaseFunction,
    forward_peaked_mu_grid,
    henyey_greenstein_pdf,
    sample_henyey_greenstein,
    tabulate_mie_phase_function,
)

ICE_M_VISIBLE = 1.3105 + 2.0e-9j


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def solver():
    return MiepythonSolver()


class TestHenyeyGreensteinPdf:
    @pytest.mark.parametrize("g", [-0.5, 0.0, 0.3, 0.89])
    def test_integrates_to_one(self, g):
        mu = np.linspace(-1.0, 1.0, 200_001)
        assert np.trapezoid(henyey_greenstein_pdf(mu, g), mu) == pytest.approx(
            1.0, rel=1e-6
        )

    @pytest.mark.parametrize("g", [-0.5, 0.0, 0.3, 0.89])
    def test_mean_cosine_recovers_the_asymmetry_parameter(self, g):
        # This is the entire point of the Henyey-Greenstein form: it is fitted
        # to reproduce g and nothing else about the true angular distribution.
        mu = np.linspace(-1.0, 1.0, 200_001)
        p = henyey_greenstein_pdf(mu, g)
        assert np.trapezoid(p * mu, mu) == pytest.approx(g, abs=1e-5)

    def test_is_flat_when_scattering_is_isotropic(self):
        mu = np.linspace(-1.0, 1.0, 11)
        assert np.allclose(henyey_greenstein_pdf(mu, 0.0), 0.5)

    def test_forward_scattering_dominates_for_positive_g(self):
        p = henyey_greenstein_pdf(np.array([-1.0, 1.0]), 0.89)
        assert p[1] > 100 * p[0]

    def test_rejects_an_unphysical_asymmetry_parameter(self):
        with pytest.raises(ValueError, match=r"\(-1, 1\)"):
            henyey_greenstein_pdf(0.0, 1.0)


class TestHenyeyGreensteinSampling:
    @pytest.mark.parametrize("g", [-0.6, 0.0, 0.45, 0.89])
    def test_sample_mean_converges_to_g(self, backend, g):
        mu = sample_henyey_greenstein(backend, backend.rng(42), g, (2_000_000,))
        # Standard error of the mean is ~1e-3 at this sample size.
        assert mu.mean() == pytest.approx(g, abs=5e-3)

    def test_samples_stay_within_the_physical_range(self, backend):
        mu = sample_henyey_greenstein(backend, backend.rng(0), 0.95, (500_000,))
        assert mu.min() >= -1.0
        assert mu.max() <= 1.0

    def test_the_same_seed_reproduces_the_same_sample(self, backend):
        first = sample_henyey_greenstein(backend, backend.rng(7), 0.8, (1000,))
        second = sample_henyey_greenstein(backend, backend.rng(7), 0.8, (1000,))
        assert np.array_equal(first, second)

    def test_the_isotropic_branch_agrees_with_the_general_one(self, backend):
        # g exactly zero takes a different code path; it must not be a
        # discontinuity in the sampled distribution.
        exact = sample_henyey_greenstein(backend, backend.rng(3), 0.0, (400_000,))
        nearly = sample_henyey_greenstein(backend, backend.rng(3), 1e-4, (400_000,))
        assert exact.mean() == pytest.approx(nearly.mean(), abs=5e-3)

    def test_the_sampled_histogram_matches_the_distribution(self, backend):
        # Compared against the exact bin probability from the closed-form CDF,
        # not against the pdf at the bin centre. The pdf varies by tens of
        # percent across the forward-most bin, so a centre-point comparison
        # measures the binning rather than the sampler.
        g = 0.7
        mu = sample_henyey_greenstein(backend, backend.rng(11), g, (2_000_000,))
        edges = np.linspace(-1.0, 1.0, 41)
        counts, _ = np.histogram(mu, bins=edges)

        def hg_cdf(m: np.ndarray) -> np.ndarray:
            return (1 - g**2) / (2 * g) * (
                1.0 / np.sqrt(1 + g**2 - 2 * g * m) - 1.0 / (1 + g)
            )

        expected = np.diff(hg_cdf(edges)) * mu.size
        assert hg_cdf(np.array([1.0]))[0] == pytest.approx(1.0)
        # Judged in units of Poisson error, because that is what actually
        # limits the comparison: the sparsest bin holds ~5e3 counts, so its
        # one-sigma spread is already 1.4%. A flat relative tolerance would
        # either fail on the tail or be vacuous on the peak, which holds a
        # hundred times more.
        deviation = np.abs(counts - expected) / np.sqrt(expected)
        assert deviation.max() < 5.0


class TestForwardPeakedGrid:
    def test_spans_the_full_cosine_range_ascending(self):
        mu = forward_peaked_mu_grid(n_points=100)
        assert np.all(np.diff(mu) > 0)
        assert mu[0] == pytest.approx(-1.0)
        assert mu[-1] == pytest.approx(1.0)

    def test_clusters_its_resolution_near_forward_scattering(self):
        mu = forward_peaked_mu_grid(n_points=1000)
        assert (1.0 - mu[-2]) < 1e-12
        # Half the grid sits within a degree of forward, which is what a
        # uniform grid cannot do at any affordable point count.
        near_forward = np.sum(mu > np.cos(np.radians(1.0)))
        assert near_forward > 0.4 * mu.size


class TestMiePhaseFunction:
    """Cross-checks between two independent routes through the solver."""

    def test_mean_cosine_agrees_with_the_efficiency_calculation(self, solver):
        # g from the phase function integral and g from the Mie series are
        # computed by entirely different code paths. Agreement is evidence
        # that the angular grid resolves what it needs to.
        m, x = ICE_M_VISIBLE, 200.0
        phase = tabulate_mie_phase_function(solver, m, x)
        _, _, g = solver.efficiencies(m, x)
        assert phase.mean_cosine == pytest.approx(float(g[0]), abs=2e-3)

    def test_normalisation_survives_a_snow_sized_grain(self, solver):
        # x ~ 1.3e3 is a 100 um grain in the visible: the regime the whole
        # project lives in.
        phase = tabulate_mie_phase_function(solver, ICE_M_VISIBLE, 1256.6)
        assert np.trapezoid(phase.p, phase.mu) == pytest.approx(1.0, rel=2e-2)

    def test_a_uniform_grid_silently_destroys_the_phase_function(self, solver):
        """Pins the trap that forward_peaked_mu_grid exists to avoid.

        A uniformly spaced mu grid steps straight over the diffraction peak.
        The failure is not subtle once measured -- the table integrates to
        roughly twenty and its mean cosine exceeds one -- but nothing raises
        at the point of use, and a transport run would simply produce wrong
        albedos. If this test ever passes, the guard has stopped working.
        """
        uniform = np.linspace(-1.0, 1.0, 20_001)
        with pytest.raises(ValueError, match="resolve the forward peak"):
            tabulate_mie_phase_function(solver, ICE_M_VISIBLE, 1256.6, mu=uniform)

    def test_rejects_the_gain_convention(self, solver):
        with pytest.raises(ValueError, match="gain"):
            solver.phase_function(1.31 - 1e-9j, 100.0, np.array([0.0, 1.0]))


class TestTabulatedSampling:
    def make_hg_table(self, g: float) -> TabulatedPhaseFunction:
        mu = np.linspace(-1.0, 1.0, 20_001)
        return TabulatedPhaseFunction(
            mu=mu, p=henyey_greenstein_pdf(mu, g), name=f"hg(g={g})"
        )

    @pytest.mark.parametrize("g", [0.0, 0.45, 0.85])
    def test_tabulated_sampling_reproduces_the_analytic_sampler(self, backend, g):
        # Two independent samplers -- closed-form inverse CDF and interpolated
        # tabulated CDF -- must agree on the distribution they draw from.
        table = self.make_hg_table(g)
        tabulated = table.sample(backend, backend.rng(5), (1_000_000,))
        analytic = sample_henyey_greenstein(backend, backend.rng(6), g, (1_000_000,))
        assert tabulated.mean() == pytest.approx(analytic.mean(), abs=5e-3)

    def test_mean_cosine_of_the_table_matches_its_g(self):
        assert self.make_hg_table(0.85).mean_cosine == pytest.approx(0.85, abs=1e-4)

    def test_samples_stay_within_the_tabulated_range(self, backend):
        mu = self.make_hg_table(0.9).sample(backend, backend.rng(1), (100_000,))
        assert mu.min() >= -1.0
        assert mu.max() <= 1.0

    def test_rejects_an_unnormalised_table(self):
        mu = np.linspace(-1.0, 1.0, 101)
        with pytest.raises(ValueError, match="integrates to"):
            TabulatedPhaseFunction(mu=mu, p=np.ones_like(mu))

    def test_rejects_a_negative_phase_function(self):
        mu = np.linspace(-1.0, 1.0, 101)
        p = henyey_greenstein_pdf(mu, 0.5)
        p[10] = -1.0
        with pytest.raises(ValueError, match="cannot be negative"):
            TabulatedPhaseFunction(mu=mu, p=p)
