"""Spectral albedo of a snowpack — the use case behind research questions 1 and 2.

Takes a configuration and the pieces it needs, returns a result object. It
decides nothing about where the numbers go: no file paths, no plotting, no
printing. That separation is what lets the same call run inside a batch job
with no display attached and lets the figures be regenerated from committed
CSVs without re-running any physics.

Two evaluation methods, and the choice is a real one:

- ``"analytic"`` — van de Hulst with similarity scaling. Accurate to about 1%,
  and fast enough to sweep a few hundred wavelengths in a second. This is what
  produces published curves.
- ``"monte-carlo"`` — the transport engine. The reference, and expensive: cost
  scales as ``1/(1 - omega)``, which for clean snow in the visible means
  millions of scattering orders per photon. Use it to spot-check the analytic
  curve at a handful of wavelengths, not to sweep one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from snow_mcrt.domain.analytic import e_folding_depth, similarity_scaled_albedo
from snow_mcrt.domain.medium import (
    BLACK_CARBON,
    MINERAL_DUST,
    ImpurityLoading,
    SnowLayer,
    compute_layer_properties,
)
from snow_mcrt.domain.optics import OpticalConstants
from snow_mcrt.domain.transport import TransportConfig, run_transport
from snow_mcrt.ports.backend import Backend
from snow_mcrt.ports.mie_solver import MieSolver

Method = Literal["analytic", "monte-carlo"]


@dataclass(frozen=True)
class SpectralAlbedoConfig:
    """Everything that determines a spectral albedo curve.

    Every field lands in the run manifest. If it is not here, it cannot have
    influenced the result.
    """

    wavelengths_nm: np.ndarray
    grain_radius_m: float = 100e-6
    snow_density: float = 300.0
    black_carbon_ng_per_g: float = 0.0
    mineral_dust_ng_per_g: float = 0.0
    method: Method = "analytic"
    transport: TransportConfig = field(default_factory=TransportConfig)
    label: str = "run"

    def as_layer(self) -> SnowLayer:
        """The snow layer this configuration describes."""
        loadings = []
        if self.black_carbon_ng_per_g:
            loadings.append(
                ImpurityLoading.from_ng_per_g(
                    BLACK_CARBON, self.black_carbon_ng_per_g
                )
            )
        if self.mineral_dust_ng_per_g:
            loadings.append(
                ImpurityLoading.from_ng_per_g(
                    MINERAL_DUST, self.mineral_dust_ng_per_g
                )
            )
        return SnowLayer(
            grain_radius_m=self.grain_radius_m,
            density=self.snow_density,
            impurities=tuple(loadings),
        )


@dataclass(frozen=True)
class SpectralAlbedoResult:
    """A spectral albedo curve with the intermediate physics that produced it.

    The single-scattering quantities are carried alongside the albedo on
    purpose. When a curve looks wrong, the first question is always whether
    the transport or the single scattering is at fault, and a result that
    reports only the albedo cannot answer it.
    """

    wavelength_nm: np.ndarray
    albedo: np.ndarray
    single_scattering_albedo: np.ndarray
    asymmetry: np.ndarray
    extinction_coefficient: np.ndarray
    e_folding_depth_m: np.ndarray
    config: SpectralAlbedoConfig
    dataset_name: str

    @property
    def co_albedo(self) -> np.ndarray:
        """``1 - omega``. The quantity that actually varies; ``omega`` itself
        sits within a millionth of unity across the visible."""
        return 1.0 - self.single_scattering_albedo

    def columns(self) -> dict[str, np.ndarray]:
        """Column name to values, in the order they are written to CSV."""
        return {
            "wavelength_nm": self.wavelength_nm,
            "albedo": self.albedo,
            "single_scattering_albedo": self.single_scattering_albedo,
            "co_albedo": self.co_albedo,
            "asymmetry": self.asymmetry,
            "extinction_coefficient_per_m": self.extinction_coefficient,
            "e_folding_depth_m": self.e_folding_depth_m,
        }


def run_spectral_albedo(
    solver: MieSolver,
    constants: OpticalConstants,
    config: SpectralAlbedoConfig,
    backend: Backend | None = None,
) -> SpectralAlbedoResult:
    """Evaluate spectral albedo over the configured wavelength grid.

    Args:
        solver: Mie solver.
        constants: Ice optical constants. Wavelengths outside its tabulated
            range raise rather than being clamped.
        config: Run parameters.
        backend: Array backend. Required only for ``method="monte-carlo"``.

    Returns:
        The curve, with the single-scattering quantities behind it.

    Raises:
        ValueError: If Monte Carlo is requested without a backend.
    """
    wavelengths = np.atleast_1d(
        np.asarray(config.wavelengths_nm, dtype=float)
    )
    m_ice = constants.m_at(wavelengths)
    layer = config.as_layer()
    props = compute_layer_properties(solver, layer, m_ice, wavelengths)

    omega = props.single_scattering_albedo
    g = props.asymmetry
    beta = props.extinction_coefficient

    if config.method == "analytic":
        albedo = similarity_scaled_albedo(omega, g)
    elif config.method == "monte-carlo":
        if backend is None:
            raise ValueError("method='monte-carlo' needs an array backend")
        albedo = np.array(
            [
                run_transport(
                    backend,
                    float(beta[i]),
                    float(omega[i]),
                    float(g[i]),
                    config=config.transport,
                ).albedo
                for i in range(wavelengths.size)
            ]
        )
    else:
        raise ValueError(f"unknown method {config.method!r}")

    return SpectralAlbedoResult(
        wavelength_nm=wavelengths,
        albedo=np.asarray(albedo, dtype=float),
        single_scattering_albedo=omega,
        asymmetry=g,
        extinction_coefficient=beta,
        e_folding_depth_m=e_folding_depth(omega, g, beta),
        config=config,
        dataset_name=constants.name,
    )
