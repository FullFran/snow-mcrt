"""Snowpack composition: ice grains, absorbing impurities, and layers.

A snow layer is an external mixture of two independent particle populations:
ice grains, which scatter almost perfectly, and absorbing impurities, which
barely scatter at all but absorb ferociously. Their cross-sections add:

.. math::
    \\beta = \\sum_i n_i \\sigma_{ext,i}, \\quad
    \\omega = \\frac{\\sum_i n_i \\sigma_{sca,i}}{\\beta}, \\quad
    g = \\frac{\\sum_i n_i \\sigma_{sca,i} g_i}{\\sum_i n_i \\sigma_{sca,i}}

*External* mixture means the impurities sit between the grains rather than
inside them. That is the assumption v1 makes and it is the conservative one:
black carbon internally mixed within an ice grain absorbs roughly twice as
strongly for the same mass, because the grain focuses light onto it. The
distinction matters for radiative forcing estimates and is out of v1 scope,
but the number this code produces is a lower bound, not an unbiased one.

Mixing ratios are quoted in ng/g throughout the literature, which is
1e-9 kg/kg. :meth:`ImpurityLoading.from_ng_per_g` is the constructor to reach
for; the raw kg/kg field exists so the arithmetic never has to carry a
conversion factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from snow_mcrt.domain.mie import ICE_DENSITY, compute_mie_properties
from snow_mcrt.ports.mie_solver import MieSolver

NG_PER_G_TO_KG_PER_KG = 1e-9


@dataclass(frozen=True)
class Impurity:
    """An absorbing particle population mixed into the snow.

    Args:
        name: Label carried into run manifests.
        radius_m: Effective particle radius in metres.
        material_density: Bulk density of the material, kg/m^3.
        refractive_index: Complex index ``n + ik``, ``k >= 0``. Held constant
            over wavelength in v1, which is defensible for black carbon across
            the visible and increasingly wrong towards the near-infrared.
    """

    name: str
    radius_m: float
    material_density: float
    refractive_index: complex

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"{self.name}: radius must be positive")
        if self.material_density <= 0:
            raise ValueError(f"{self.name}: material density must be positive")
        if self.refractive_index.imag < 0:
            raise ValueError(
                f"{self.name}: the convention is m = n + ik with k >= 0; a "
                f"negative imaginary part describes gain, not absorption"
            )

    @property
    def particle_mass(self) -> float:
        """Mass of a single particle, kg."""
        return self.material_density * (4.0 / 3.0) * math.pi * self.radius_m**3


# Bond & Bergstrom (2006) for the black carbon index; the radius is the
# accumulation-mode value used throughout the snow-impurity literature.
BLACK_CARBON = Impurity(
    name="black carbon",
    radius_m=0.05e-6,
    material_density=1800.0,
    refractive_index=1.95 + 0.79j,
)

# Mineral dust is far weaker per unit mass and coarser, so it takes hundreds
# of times the loading to darken snow as much as black carbon does.
MINERAL_DUST = Impurity(
    name="mineral dust",
    radius_m=1.0e-6,
    material_density=2600.0,
    refractive_index=1.55 + 0.0015j,
)


@dataclass(frozen=True)
class ImpurityLoading:
    """An impurity together with how much of it is present.

    Args:
        impurity: The particle population.
        mass_mixing_ratio: Mass of impurity per mass of snow, kg/kg.
    """

    impurity: Impurity
    mass_mixing_ratio: float

    def __post_init__(self) -> None:
        if self.mass_mixing_ratio < 0:
            raise ValueError("mass mixing ratio must be non-negative")
        if self.mass_mixing_ratio > 1:
            raise ValueError("mass mixing ratio cannot exceed 1 kg/kg")

    @classmethod
    def from_ng_per_g(cls, impurity: Impurity, ng_per_g: float) -> ImpurityLoading:
        """Build from the ng/g the literature quotes."""
        return cls(impurity, ng_per_g * NG_PER_G_TO_KG_PER_KG)

    @property
    def ng_per_g(self) -> float:
        """Mixing ratio in ng/g."""
        return self.mass_mixing_ratio / NG_PER_G_TO_KG_PER_KG

    def number_density(self, snow_density: float) -> float:
        """Particles per cubic metre in snow of the given bulk density."""
        return self.mass_mixing_ratio * snow_density / self.impurity.particle_mass


@dataclass(frozen=True)
class SnowLayer:
    """A homogeneous layer of snow.

    Args:
        grain_radius_m: Ice grain radius in metres.
        density: Bulk snow density, kg/m^3.
        thickness_m: Layer thickness in metres. ``inf`` for a semi-infinite
            pack, which is what the analytic oracle assumes.
        impurities: Absorbing populations mixed in.
    """

    grain_radius_m: float
    density: float
    thickness_m: float = math.inf
    impurities: tuple[ImpurityLoading, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.grain_radius_m <= 0:
            raise ValueError("grain radius must be positive")
        if not 0 < self.density <= ICE_DENSITY:
            raise ValueError(
                f"snow density must lie in (0, {ICE_DENSITY}] kg/m^3, "
                f"got {self.density}"
            )
        if self.thickness_m <= 0:
            raise ValueError("layer thickness must be positive")

    @property
    def grain_number_density(self) -> float:
        """Ice grains per cubic metre."""
        grain_mass = ICE_DENSITY * (4.0 / 3.0) * math.pi * self.grain_radius_m**3
        return self.density / grain_mass


@dataclass(frozen=True)
class LayerOpticalProperties:
    """What the transport loop consumes for one layer, per wavelength.

    Args:
        wavelength_nm: Wavelength grid.
        extinction_coefficient: ``beta``, m^-1.
        single_scattering_albedo: ``omega``.
        asymmetry: ``g``.
        thickness_m: Layer thickness, carried through for optical depth.
    """

    wavelength_nm: np.ndarray
    extinction_coefficient: np.ndarray
    single_scattering_albedo: np.ndarray
    asymmetry: np.ndarray
    thickness_m: float

    @property
    def optical_depth(self) -> np.ndarray:
        """``beta * thickness``. Infinite for a semi-infinite layer."""
        if math.isinf(self.thickness_m):
            return np.full_like(self.extinction_coefficient, np.inf)
        return self.extinction_coefficient * self.thickness_m


def compute_layer_properties(
    solver: MieSolver,
    layer: SnowLayer,
    m_ice: np.ndarray | complex,
    wavelength_nm: np.ndarray | float,
) -> LayerOpticalProperties:
    """Combine ice grains and impurities into bulk optical properties.

    Args:
        solver: Mie solver.
        layer: The snow layer.
        m_ice: Complex index of ice ``n + ik`` at each wavelength.
        wavelength_nm: Wavelength grid in nanometres.

    Returns:
        Extinction coefficient, single-scattering albedo and asymmetry
        parameter on the wavelength grid.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))

    ice = compute_mie_properties(
        solver, m_ice, layer.grain_radius_m, wavelengths
    )
    n_ice = layer.grain_number_density
    beta_ext = n_ice * ice.cross_section_ext
    beta_sca = n_ice * ice.cross_section_sca
    beta_sca_g = beta_sca * ice.g

    for loading in layer.impurities:
        particle = loading.impurity
        props = compute_mie_properties(
            solver, particle.refractive_index, particle.radius_m, wavelengths
        )
        n_imp = loading.number_density(layer.density)
        beta_ext = beta_ext + n_imp * props.cross_section_ext
        sca = n_imp * props.cross_section_sca
        beta_sca = beta_sca + sca
        beta_sca_g = beta_sca_g + sca * props.g

    omega = beta_sca / beta_ext
    # A layer with no scattering at all has no defined mean scattering angle.
    # Reporting zero there is honest: it is never consulted, because the
    # transport loop only samples an angle once it has decided to scatter.
    g = np.divide(
        beta_sca_g, beta_sca, out=np.zeros_like(beta_sca), where=beta_sca > 0
    )

    return LayerOpticalProperties(
        wavelength_nm=wavelengths,
        extinction_coefficient=beta_ext,
        single_scattering_albedo=omega,
        asymmetry=g,
        thickness_m=layer.thickness_m,
    )
