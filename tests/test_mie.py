"""Ground truth: analytic limits of Mie theory, and the snow regime.

Organised by what each test compares against, not by which module it touches.
The Rayleigh and geometric limits are closed-form, so they are the oracle here
in the same sense that Onsager's solution is the oracle for a 2-D Ising model:
a solver that reproduces both is constrained at each end of the size range,
and the interval between them is where the series does the real work.
"""

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.domain.mie import ICE_DENSITY, MieProperties, compute_mie_properties

# Ice in the mid-visible. The real part is flat at ~1.31 across the visible;
# the imaginary part is the smallest of any natural material near 470 nm.
ICE_M_VISIBLE = 1.3105 + 2.0e-9j


@pytest.fixture
def solver():
    return MiepythonSolver()


def rayleigh_q_sca(m: complex, x: float) -> float:
    """Closed-form scattering efficiency in the small-particle limit."""
    y = (m**2 - 1.0) / (m**2 + 2.0)
    return (8.0 / 3.0) * x**4 * abs(y) ** 2


class TestRayleighLimit:
    def test_matches_the_closed_form_for_a_transparent_sphere(self, solver):
        m, x = 1.33 + 0j, 0.01
        q_ext, q_sca, _ = solver.efficiencies(m, x)
        assert q_sca[0] == pytest.approx(rayleigh_q_sca(m, x), rel=1e-4)

    def test_scattering_scales_as_the_fourth_power_of_size(self, solver):
        x = np.array([1e-3, 3e-3, 1e-2])
        _, q_sca, _ = solver.efficiencies(1.33 + 0j, x)
        slope = np.polyfit(np.log(x), np.log(q_sca), 1)[0]
        assert slope == pytest.approx(4.0, abs=1e-3)

    def test_small_spheres_scatter_nearly_isotropically(self, solver):
        _, _, g = solver.efficiencies(1.33 + 0j, 0.01)
        assert abs(g[0]) < 1e-3


class TestGeometricLimit:
    def test_extinction_approaches_twice_the_geometric_cross_section(self, solver):
        # The extinction paradox: a large sphere removes twice the light its
        # shadow would suggest, the surplus being forward diffraction.
        q_ext, _, _ = solver.efficiencies(1.31 + 0j, 1.0e4)
        assert q_ext[0] == pytest.approx(2.0, abs=0.05)

    def test_large_spheres_scatter_strongly_forward(self, solver):
        _, _, g = solver.efficiencies(1.31 + 0j, 1.0e4)
        assert 0.8 < g[0] < 1.0


class TestEnergyConservation:
    def test_scattering_never_exceeds_extinction(self, solver):
        x = np.logspace(-2, 4, 40)
        q_ext, q_sca, _ = solver.efficiencies(ICE_M_VISIBLE, x)
        assert np.all(q_sca <= q_ext * (1 + 1e-12))

    def test_a_transparent_sphere_absorbs_nothing(self, solver):
        props = compute_mie_properties(solver, 1.31 + 0j, 100e-6, 500.0)
        assert np.all(props.q_abs < 1e-12)
        assert props.single_scattering_albedo[0] == pytest.approx(1.0)

    def test_an_absorbing_sphere_absorbs_something(self, solver):
        props = compute_mie_properties(solver, 1.31 + 1e-4j, 100e-6, 500.0)
        assert props.q_abs[0] > 0
        assert props.single_scattering_albedo[0] < 1.0


class TestSignConvention:
    """The conjugation at the adapter boundary is load-bearing.

    ``miepython`` takes ``m = n - ik``; the domain speaks ``m = n + ik``. If
    the adapter ever stops conjugating, absorption turns into gain. Nothing
    raises -- ``Q_sca`` merely creeps above ``Q_ext`` by a part in ten
    thousand, which no plot would ever show.
    """

    def test_positive_k_produces_absorption_not_gain(self, solver):
        q_ext, q_sca, _ = solver.efficiencies(1.31 + 1e-3j, 100.0)
        assert q_ext[0] > q_sca[0]

    def test_more_absorbing_ice_absorbs_more(self, solver):
        weak = compute_mie_properties(solver, 1.31 + 1e-8j, 100e-6, 500.0)
        strong = compute_mie_properties(solver, 1.31 + 1e-5j, 100e-6, 500.0)
        assert strong.q_abs[0] > weak.q_abs[0]

    def test_the_domain_convention_is_enforced_at_the_boundary(self, solver):
        with pytest.raises(ValueError, match="gain"):
            solver.efficiencies(1.31 - 1e-8j, 100.0)


class TestSnowRegime:
    """Values a snow-optics reader should recognise on sight."""

    def test_clean_snow_in_the_visible_is_a_near_perfect_scatterer(self, solver):
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        co_albedo = 1.0 - props.single_scattering_albedo[0]
        # Wiscombe & Warren (1980): 1 - omega is of order 1e-7 to 1e-6 here.
        # This is why visible snow albedo is set by trace absorbers rather
        # than by the ice itself.
        assert 0 < co_albedo < 1e-4

    def test_the_asymmetry_parameter_sits_near_the_canonical_value(self, solver):
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        assert 0.85 < props.g[0] < 0.92

    def test_absorption_grows_from_the_visible_into_the_near_infrared(self, solver):
        # Ice absorption climbs by orders of magnitude towards 1300 nm, which
        # is what makes near-infrared albedo the grain-size signal.
        visible = compute_mie_properties(solver, 1.3105 + 2.0e-9j, 100e-6, 500.0)
        near_ir = compute_mie_properties(solver, 1.2985 + 1.3e-5j, 100e-6, 1300.0)
        assert near_ir.q_abs[0] > visible.q_abs[0]
        assert (
            near_ir.single_scattering_albedo[0]
            < visible.single_scattering_albedo[0]
        )

    def test_larger_grains_absorb_more_of_what_they_intercept(self, solver):
        # The grain-size signal: bigger grains mean longer path in ice per
        # scattering event, so a lower single-scattering albedo.
        fine = compute_mie_properties(solver, 1.2985 + 1.3e-5j, 50e-6, 1300.0)
        coarse = compute_mie_properties(solver, 1.2985 + 1.3e-5j, 1000e-6, 1300.0)
        assert (
            coarse.single_scattering_albedo[0]
            < fine.single_scattering_albedo[0]
        )


class TestBulkProperties:
    def test_extinction_coefficient_scales_with_number_density(self, solver):
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        single = props.extinction_coefficient(1e6)
        double = props.extinction_coefficient(2e6)
        assert double[0] == pytest.approx(2 * single[0])

    def test_extinction_length_in_seasonal_snow_is_millimetric(self, solver):
        # 300 kg/m^3, 100 um grains: the mean free path should come out in
        # millimetres. Metres or micrometres would both indicate a units slip
        # in the density-to-number-density conversion.
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        beta = props.extinction_coefficient_from_density(300.0)[0]
        mean_free_path_mm = 1e3 / beta
        assert 0.1 < mean_free_path_mm < 10.0

    def test_density_cannot_exceed_solid_ice(self, solver):
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        with pytest.raises(ValueError, match="exceeds solid ice"):
            props.extinction_coefficient_from_density(ICE_DENSITY + 1.0)

    def test_broadcasts_over_a_wavelength_grid(self, solver):
        wavelengths = np.linspace(400.0, 700.0, 16)
        props = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, wavelengths)
        assert props.q_ext.shape == (16,)
        assert props.single_scattering_albedo.shape == (16,)


class TestMiePropertiesInvariants:
    def test_rejects_scattering_above_extinction(self):
        with pytest.raises(ValueError, match="negative absorption"):
            MieProperties(
                x=np.array([1.0]),
                q_ext=np.array([1.0]),
                q_sca=np.array([1.5]),
                g=np.array([0.0]),
                radius_m=np.array([1e-6]),
            )

    def test_rejects_an_out_of_range_asymmetry_parameter(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            MieProperties(
                x=np.array([1.0]),
                q_ext=np.array([2.0]),
                q_sca=np.array([1.0]),
                g=np.array([1.5]),
                radius_m=np.array([1e-6]),
            )
