"""Snowpack composition: mixing, impurities, and the darkening they cause."""

import math

import numpy as np
import pytest

from snow_mcrt.adapters.miepython_solver import MiepythonSolver
from snow_mcrt.domain.analytic import semi_infinite_albedo
from snow_mcrt.domain.medium import (
    BLACK_CARBON,
    MINERAL_DUST,
    Impurity,
    ImpurityLoading,
    SnowLayer,
    compute_layer_properties,
)
from snow_mcrt.domain.mie import LogNormalGrainSizes, compute_mie_properties

ICE_M_VISIBLE = 1.3105 + 2.0e-9j
ICE_M_NEAR_IR = 1.2985 + 1.3e-5j


@pytest.fixture
def solver():
    return MiepythonSolver()


def layer_albedo(solver, layer, m_ice, wavelength_nm) -> float:
    props = compute_layer_properties(solver, layer, m_ice, wavelength_nm)
    return float(
        semi_infinite_albedo(
            props.single_scattering_albedo[0], props.asymmetry[0]
        )
    )


def with_black_carbon(ng_per_g: float, **kwargs) -> SnowLayer:
    loadings = (
        (ImpurityLoading.from_ng_per_g(BLACK_CARBON, ng_per_g),)
        if ng_per_g
        else ()
    )
    return SnowLayer(
        grain_radius_m=kwargs.pop("grain_radius_m", 100e-6),
        density=kwargs.pop("density", 300.0),
        impurities=loadings,
        **kwargs,
    )


class TestMixingRatios:
    def test_ng_per_g_round_trips(self):
        loading = ImpurityLoading.from_ng_per_g(BLACK_CARBON, 250.0)
        assert loading.ng_per_g == pytest.approx(250.0)
        assert loading.mass_mixing_ratio == pytest.approx(250e-9)

    def test_number_density_scales_with_the_mixing_ratio(self):
        one = ImpurityLoading.from_ng_per_g(BLACK_CARBON, 100.0)
        two = ImpurityLoading.from_ng_per_g(BLACK_CARBON, 200.0)
        assert two.number_density(300.0) == pytest.approx(
            2 * one.number_density(300.0)
        )

    def test_a_hundred_nanograms_per_gram_is_a_lot_of_particles(self):
        # ~2e13 particles per cubic metre, against ~7e8 ice grains. Impurities
        # outnumber grains by four orders of magnitude and still amount to a
        # tenth of a part per million by mass.
        loading = ImpurityLoading.from_ng_per_g(BLACK_CARBON, 100.0)
        assert 1e12 < loading.number_density(300.0) < 1e15

    def test_rejects_a_negative_mixing_ratio(self):
        with pytest.raises(ValueError, match="non-negative"):
            ImpurityLoading(BLACK_CARBON, -1e-9)


class TestPureSnow:
    def test_a_layer_with_no_impurities_is_just_its_grains(self, solver):
        # The mixture machinery must be an identity when there is nothing to
        # mix, or every impurity result is offset by an unknown constant.
        # Monodisperse on purpose: this compares against a single-radius Mie
        # call, so the size distribution has to be switched off to make the
        # two comparable.
        layer = with_black_carbon(0.0, grain_sigma_g=1.0)
        props = compute_layer_properties(solver, layer, ICE_M_VISIBLE, 500.0)
        grains = compute_mie_properties(solver, ICE_M_VISIBLE, 100e-6, 500.0)
        assert props.single_scattering_albedo[0] == pytest.approx(
            grains.single_scattering_albedo[0]
        )
        assert props.asymmetry[0] == pytest.approx(grains.g[0])
        assert props.extinction_coefficient[0] == pytest.approx(
            grains.extinction_coefficient_from_density(300.0)[0]
        )

    def test_extinction_scales_with_snow_density(self, solver):
        thin = compute_layer_properties(
            solver, SnowLayer(100e-6, 150.0), ICE_M_VISIBLE, 500.0
        )
        dense = compute_layer_properties(
            solver, SnowLayer(100e-6, 300.0), ICE_M_VISIBLE, 500.0
        )
        assert dense.extinction_coefficient[0] == pytest.approx(
            2 * thin.extinction_coefficient[0]
        )

    def test_density_does_not_change_the_albedo_of_a_deep_pack(self, solver):
        # Density sets how fast light is extinguished, not what fraction is
        # absorbed per event. A semi-infinite pack has no bottom to reach, so
        # its albedo is density-independent. Worth pinning: an implementation
        # that let density leak into omega would still look plausible.
        assert layer_albedo(
            solver, SnowLayer(100e-6, 150.0), ICE_M_VISIBLE, 500.0
        ) == pytest.approx(
            layer_albedo(solver, SnowLayer(100e-6, 400.0), ICE_M_VISIBLE, 500.0)
        )


class TestBlackCarbon:
    def test_darkens_the_visible_monotonically(self, solver):
        albedos = [
            layer_albedo(solver, with_black_carbon(ng), ICE_M_VISIBLE, 500.0)
            for ng in (0.0, 1.0, 10.0, 100.0, 1000.0)
        ]
        assert np.all(np.diff(albedos) < 0)

    def test_reproduces_the_published_magnitude(self, solver):
        # Warren & Wiscombe (1980) Fig. 2: 100 ng/g of soot in snow with
        # ~100 um grains drops visible albedo from ~0.99 to the mid 0.95s.
        # This is research question 2, checked here at a single point before
        # the full curve reproduction.
        clean = layer_albedo(solver, with_black_carbon(0.0), ICE_M_VISIBLE, 500.0)
        dirty = layer_albedo(solver, with_black_carbon(100.0), ICE_M_VISIBLE, 500.0)
        assert clean > 0.98
        assert 0.94 < dirty < 0.97

    def test_a_part_per_billion_is_already_measurable(self, solver):
        # The whole reason trace impurities matter: ice is so weakly absorbing
        # in the visible that 1 ng/g of carbon competes with it.
        clean = layer_albedo(solver, with_black_carbon(0.0), ICE_M_VISIBLE, 500.0)
        trace = layer_albedo(solver, with_black_carbon(1.0), ICE_M_VISIBLE, 500.0)
        assert 1e-4 < clean - trace < 1e-2

    def test_barely_touches_the_asymmetry_parameter(self, solver):
        # Impurities absorb, they do not meaningfully scatter. If g moved
        # here, the cross-section weighting in the mixture would be wrong.
        clean = compute_layer_properties(
            solver, with_black_carbon(0.0), ICE_M_VISIBLE, 500.0
        )
        dirty = compute_layer_properties(
            solver, with_black_carbon(1000.0), ICE_M_VISIBLE, 500.0
        )
        assert dirty.asymmetry[0] == pytest.approx(clean.asymmetry[0], abs=1e-3)

    def test_matters_far_less_where_ice_already_absorbs(self, solver):
        # At 1300 nm the ice itself absorbs four orders of magnitude more
        # strongly, so carbon has almost nothing left to compete with. This
        # is why impurity retrieval works in the visible and not the
        # near-infrared.
        visible_drop = layer_albedo(
            solver, with_black_carbon(0.0), ICE_M_VISIBLE, 500.0
        ) - layer_albedo(solver, with_black_carbon(500.0), ICE_M_VISIBLE, 500.0)
        near_ir_drop = layer_albedo(
            solver, with_black_carbon(0.0), ICE_M_NEAR_IR, 1300.0
        ) - layer_albedo(solver, with_black_carbon(500.0), ICE_M_NEAR_IR, 1300.0)
        assert visible_drop > 10 * near_ir_drop


class TestMineralDust:
    def test_darkens_the_visible_far_more_weakly_than_carbon(self, solver):
        # Dust is roughly two orders of magnitude less effective per unit
        # mass, which is why field studies report it in ppm and carbon in ppb.
        def albedo(impurity, ng):
            layer = SnowLayer(
                100e-6,
                300.0,
                impurities=(ImpurityLoading.from_ng_per_g(impurity, ng),),
            )
            return layer_albedo(solver, layer, ICE_M_VISIBLE, 500.0)

        clean = layer_albedo(solver, SnowLayer(100e-6, 300.0), ICE_M_VISIBLE, 500.0)
        carbon_drop = clean - albedo(BLACK_CARBON, 1000.0)
        dust_drop = clean - albedo(MINERAL_DUST, 1000.0)
        assert dust_drop > 0
        assert carbon_drop > 20 * dust_drop

    def test_enough_dust_still_darkens_snow(self, solver):
        layer = SnowLayer(
            100e-6,
            300.0,
            impurities=(ImpurityLoading.from_ng_per_g(MINERAL_DUST, 100_000.0),),
        )
        assert layer_albedo(solver, layer, ICE_M_VISIBLE, 500.0) < 0.95


class TestCombinedLoadings:
    def test_two_impurities_darken_more_than_either_alone(self, solver):
        def albedo(loadings):
            return layer_albedo(
                solver,
                SnowLayer(100e-6, 300.0, impurities=loadings),
                ICE_M_VISIBLE,
                500.0,
            )

        carbon = ImpurityLoading.from_ng_per_g(BLACK_CARBON, 100.0)
        dust = ImpurityLoading.from_ng_per_g(MINERAL_DUST, 100_000.0)
        both = albedo((carbon, dust))
        assert both < albedo((carbon,))
        assert both < albedo((dust,))


class TestOpticalDepth:
    def test_is_extinction_times_thickness(self, solver):
        layer = SnowLayer(100e-6, 300.0, thickness_m=0.05)
        props = compute_layer_properties(solver, layer, ICE_M_VISIBLE, 500.0)
        assert props.optical_depth[0] == pytest.approx(
            props.extinction_coefficient[0] * 0.05
        )

    def test_a_semi_infinite_layer_has_infinite_optical_depth(self, solver):
        props = compute_layer_properties(
            solver, SnowLayer(100e-6, 300.0), ICE_M_VISIBLE, 500.0
        )
        assert np.all(np.isinf(props.optical_depth))

    def test_a_five_centimetre_layer_is_already_optically_deep(self, solver):
        # Mean free path is ~0.2 mm, so 5 cm is hundreds of optical depths.
        # Any transport result should be indistinguishable from semi-infinite
        # there, which is what makes the analytic oracle applicable at all.
        layer = SnowLayer(100e-6, 300.0, thickness_m=0.05)
        props = compute_layer_properties(solver, layer, ICE_M_VISIBLE, 500.0)
        assert props.optical_depth[0] > 100


class TestBroadcasting:
    def test_evaluates_over_a_wavelength_grid(self, solver):
        wavelengths = np.linspace(400.0, 700.0, 12)
        m_ice = np.full(wavelengths.shape, ICE_M_VISIBLE)
        props = compute_layer_properties(
            solver, with_black_carbon(100.0), m_ice, wavelengths
        )
        assert props.single_scattering_albedo.shape == (12,)
        assert props.extinction_coefficient.shape == (12,)
        assert props.asymmetry.shape == (12,)


class TestValidation:
    def test_rejects_snow_denser_than_ice(self):
        with pytest.raises(ValueError, match="snow density must lie"):
            SnowLayer(100e-6, 1000.0)

    def test_rejects_a_nonpositive_grain_radius(self):
        with pytest.raises(ValueError, match="grain radius must be positive"):
            SnowLayer(0.0, 300.0)

    def test_rejects_a_nonpositive_thickness(self):
        with pytest.raises(ValueError, match="thickness must be positive"):
            SnowLayer(100e-6, 300.0, thickness_m=0.0)

    def test_rejects_the_gain_convention_on_an_impurity(self):
        with pytest.raises(ValueError, match="gain"):
            Impurity("bad", 1e-7, 1800.0, 1.95 - 0.79j)

    def test_a_semi_infinite_layer_is_the_default(self):
        assert math.isinf(SnowLayer(100e-6, 300.0).thickness_m)


class TestGrainSizeDistribution:
    """Monodisperse spheres are an actively misleading idealisation.

    A perfect sphere supports morphology-dependent resonances -- whispering
    gallery modes -- that spike absorption by more than tenfold at isolated
    size parameters. They are real physics for one sphere and pure fiction for
    a snowpack, and they show up in a spectral albedo curve as spikes no real
    snow exhibits. Averaging over a size distribution is the fix, and it is
    why the default is 1.5 rather than 1.
    """

    RESONANT_WAVELENGTH_NM = 676.692468
    RESONANT_M = 1.30739922596 + 2.023849359999999e-08j

    def co_albedo(self, solver, sigma_g: float) -> float:
        layer = SnowLayer(100e-6, 300.0, grain_sigma_g=sigma_g)
        props = compute_layer_properties(
            solver, layer, self.RESONANT_M, self.RESONANT_WAVELENGTH_NM
        )
        return float(1.0 - props.single_scattering_albedo[0])

    def test_a_monodisperse_pack_shows_the_resonance(self, solver):
        # Pins the defect the default exists to avoid. Measured against the
        # smooth background of ~4e-5, this point sits five times higher.
        assert self.co_albedo(solver, 1.0) > 1.5e-4

    def test_a_realistic_distribution_removes_it(self, solver):
        # The smooth neighbours of this wavelength sit at 4.4e-5 and 5.5e-5,
        # so anything in that band is the resonance genuinely gone rather than
        # merely reduced.
        assert self.co_albedo(solver, 1.5) < 5.5e-5

    def test_an_odd_node_count_is_refused(self):
        # An odd, symmetric quadrature puts a node exactly on the median
        # radius with the largest weight in the distribution. At 676.7 nm that
        # radius is the resonant one, so the resonance survives the averaging.
        # Measured: n=17 gives 6.6e-5 and n=33 gives 5.7e-5, against 4.7e-5
        # for every even count from 16 to 64.
        with pytest.raises(ValueError, match="must be even"):
            LogNormalGrainSizes(100e-6, sigma_g=1.5, n_quadrature=17)

    def test_the_default_node_count_is_even(self):
        assert LogNormalGrainSizes(100e-6).n_quadrature % 2 == 0

    def test_the_result_is_converged_in_the_node_count(self, solver):
        # If doubling the quadrature moved the answer, the smoothing would be
        # an artefact of the sampling rather than a property of the medium.
        coarse = LogNormalGrainSizes(100e-6, 1.5, n_quadrature=16)
        fine = LogNormalGrainSizes(100e-6, 1.5, n_quadrature=32)
        assert fine.mean_cube_radius == pytest.approx(
            coarse.mean_cube_radius, rel=0.02
        )

    def test_even_a_five_percent_spread_removes_it(self, solver):
        # Real snow is nowhere near this narrow. If a 5% spread already
        # suffices, no physical snowpack can display these features.
        assert self.co_albedo(solver, 1.05) < 7e-5

    def test_the_default_is_polydisperse(self):
        assert SnowLayer(100e-6, 300.0).grain_sigma_g == 1.5

    def test_a_monodisperse_distribution_is_a_single_radius(self):
        radii, weights = LogNormalGrainSizes(100e-6, sigma_g=1.0).radii_and_weights()
        assert radii.size == 1
        assert radii[0] == pytest.approx(100e-6)
        assert weights[0] == pytest.approx(1.0)

    def test_weights_are_normalised(self):
        _, weights = LogNormalGrainSizes(100e-6, sigma_g=1.6).radii_and_weights()
        assert weights.sum() == pytest.approx(1.0)

    def test_mean_cube_radius_matches_the_lognormal_moment(self, solver):
        # Number density is fixed by mass, so <r^3> is what enters, not the
        # median cubed. For a log-normal those differ by exp(4.5 ln^2 sigma) --
        # a factor of 2.1 at sigma_g = 1.5. Using the median would overcount
        # grains and inflate extinction by the same factor.
        sizes = LogNormalGrainSizes(100e-6, sigma_g=1.5)
        expected = (100e-6) ** 3 * np.exp(4.5 * np.log(1.5) ** 2)
        assert sizes.mean_cube_radius == pytest.approx(expected, rel=1e-3)
        assert sizes.mean_cube_radius > 2 * (100e-6) ** 3

    def test_broader_distributions_change_extinction(self, solver):
        narrow = compute_layer_properties(
            solver, SnowLayer(100e-6, 300.0, grain_sigma_g=1.0), ICE_M_VISIBLE, 500.0
        )
        broad = compute_layer_properties(
            solver, SnowLayer(100e-6, 300.0, grain_sigma_g=1.5), ICE_M_VISIBLE, 500.0
        )
        assert broad.extinction_coefficient[0] != narrow.extinction_coefficient[0]

    def test_rejects_a_distribution_narrower_than_a_point(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            SnowLayer(100e-6, 300.0, grain_sigma_g=0.9)
