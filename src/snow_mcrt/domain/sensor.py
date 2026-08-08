"""Spectral response of a satellite instrument, and integration over it.

A radiative transfer model produces albedo as a function of wavelength. A
satellite does not measure that. It measures a handful of numbers, each one an
integral of the scene's reflectance against the spectral response of a
detector. Turning the first into the second is the step that makes a physics
model comparable with an image, and it is not a detail: the same snowpack
reports different numbers to Sentinel-2 and to MODIS because their bands are
different, and neither number equals the albedo at the band's centre.

The quantity a band reports is

.. math::
    \\alpha_b = \\frac{\\int \\alpha(\\lambda)\\, R_b(\\lambda)\\, E(\\lambda)\\, d\\lambda}
                     {\\int R_b(\\lambda)\\, E(\\lambda)\\, d\\lambda}

with :math:`R_b` the band's spectral response and :math:`E` the incident
spectral irradiance. **Both weights matter and they are not interchangeable.**
The response says what the instrument can see; the irradiance says how much
light there was to see it with. Dropping :math:`E` is a defensible
approximation for a narrow band, where the solar spectrum barely moves across
it, and a poor one for a wide band in the near infrared, where it moves a lot.
:func:`band_albedo` therefore takes the irradiance as an explicit optional
argument rather than silently assuming it flat.

**On top-hat responses.** A real response function is a measured curve with
sloped shoulders and out-of-band leakage, distributed by the agency that flies
the instrument. :class:`TopHat` is the rectangular idealisation, and it is what
this module ships with because it needs no external file. It is an
approximation, it is named as one, and :class:`Tabulated` exists so a measured
curve can replace it without touching anything else.

References:
    Dozier, J. (1989). Spectral signature of alpine snow cover from the Landsat
        Thematic Mapper. *Remote Sensing of Environment*, 28, 9-22.
    Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission
        for GMES operational services. *Remote Sensing of Environment*, 120, 25-36.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    """One spectral band: a name and the response that defines it.

    ``response`` is evaluated on whatever wavelength grid the caller has. It
    need not be normalised -- the integration divides by its own weight, so
    only the shape matters.
    """

    name: str
    response: "SpectralResponse"

    @property
    def centre_nm(self) -> float:
        return self.response.centre_nm


class SpectralResponse:
    """A band's sensitivity as a function of wavelength."""

    centre_nm: float

    def __call__(self, wavelength_nm: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class TopHat(SpectralResponse):
    """Rectangular response: unity inside the band, zero outside.

    The idealisation every band table implies when it quotes a centre and a
    width. It is exact for nothing and adequate for a lot: for snow, whose
    spectral albedo is smooth over the width of a typical band, the error it
    introduces is small in the visible and grows in the near infrared, where
    ice absorption turns sharply and the curve is no longer locally linear.

    Use :class:`Tabulated` with a measured response when that error matters.
    :func:`snow_mcrt.application.band_albedo.response_shape_sensitivity`
    quantifies it rather than asserting it is negligible.
    """

    centre_nm: float
    width_nm: float

    def __call__(self, wavelength_nm: np.ndarray) -> np.ndarray:
        half = self.width_nm / 2.0
        inside = np.abs(np.asarray(wavelength_nm) - self.centre_nm) <= half
        return inside.astype(float)


@dataclass(frozen=True)
class Gaussian(SpectralResponse):
    """Gaussian response, truncated at three standard deviations.

    Not a claim about any real instrument. It exists so the effect of the
    response *shape* can be measured against :class:`TopHat` at matched
    equivalent width, separating "which band" from "what shape of band".
    """

    centre_nm: float
    fwhm_nm: float

    def __call__(self, wavelength_nm: np.ndarray) -> np.ndarray:
        sigma = self.fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        x = (np.asarray(wavelength_nm) - self.centre_nm) / sigma
        return np.where(np.abs(x) <= 3.0, np.exp(-0.5 * x * x), 0.0)


@dataclass(frozen=True)
class Tabulated(SpectralResponse):
    """A measured response curve, interpolated onto the caller's grid.

    This is the adapter for a real instrument. Agencies distribute these as
    tables; drop one in here and every band quantity in the repository is
    computed against the instrument that actually flew.
    """

    centre_nm: float
    wavelength_nm: np.ndarray
    response: np.ndarray

    def __call__(self, wavelength_nm: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(wavelength_nm),
            self.wavelength_nm,
            self.response,
            left=0.0,
            right=0.0,
        )


def band_albedo(
    wavelength_nm: np.ndarray,
    albedo: np.ndarray,
    band: Band,
    irradiance: np.ndarray | None = None,
) -> float:
    """Response-weighted albedo for one band.

    Args:
        wavelength_nm: Grid the albedo is given on, ascending.
        albedo: Spectral albedo on that grid.
        band: The band to integrate over.
        irradiance: Incident spectral irradiance on the same grid. When
            omitted the weighting is by response alone, which assumes the
            source is flat across the band. That assumption is reasonable for
            a narrow visible band and progressively worse for a wide one in
            the near infrared.

    Returns:
        The band albedo.

    Raises:
        ValueError: If the band lies outside the wavelength grid, so that the
            weight integrates to zero. Returning a silent ``nan`` here would
            let a band that was never measured travel downstream as if it had
            been.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    albedo = np.asarray(albedo, dtype=float)

    weight = band.response(wavelength_nm)
    if irradiance is not None:
        weight = weight * np.asarray(irradiance, dtype=float)

    denominator = np.trapezoid(weight, wavelength_nm)
    if denominator <= 0.0:
        raise ValueError(
            f"Band {band.name!r} (centre {band.centre_nm} nm) has no overlap "
            f"with the wavelength grid {wavelength_nm[0]:.0f}-"
            f"{wavelength_nm[-1]:.0f} nm."
        )

    return float(np.trapezoid(albedo * weight, wavelength_nm) / denominator)


# ---------------------------------------------------------------------------
# Nominal band definitions
#
# Centres and widths as published in the mission specifications. They are
# top-hat idealisations of measured curves -- see the module docstring.
# ---------------------------------------------------------------------------

#: Sentinel-2 MSI, the bands that see snow. Drusch et al. (2012).
SENTINEL2_MSI: dict[str, Band] = {
    "B2": Band("B2", TopHat(490.0, 65.0)),
    "B3": Band("B3", TopHat(560.0, 35.0)),
    "B4": Band("B4", TopHat(665.0, 30.0)),
    "B8": Band("B8", TopHat(842.0, 115.0)),
    "B8A": Band("B8A", TopHat(865.0, 20.0)),
    "B11": Band("B11", TopHat(1610.0, 90.0)),
    "B12": Band("B12", TopHat(2190.0, 180.0)),
}

#: MODIS land bands 1-7. Used by the operational snow products.
MODIS_LAND: dict[str, Band] = {
    "b1": Band("b1", TopHat(645.0, 50.0)),
    "b2": Band("b2", TopHat(858.5, 35.0)),
    "b3": Band("b3", TopHat(469.0, 20.0)),
    "b4": Band("b4", TopHat(555.0, 20.0)),
    "b5": Band("b5", TopHat(1240.0, 20.0)),
    "b6": Band("b6", TopHat(1640.0, 24.0)),
    "b7": Band("b7", TopHat(2130.0, 50.0)),
}

#: The band pair each instrument uses for the snow index. See :func:`ndsi`.
NDSI_BANDS: dict[str, tuple[str, str]] = {
    "sentinel2": ("B3", "B11"),
    "modis": ("b4", "b6"),
}


def ndsi(visible: float, shortwave_infrared: float) -> float:
    """Normalised Difference Snow Index.

    .. math::
        \\mathrm{NDSI} = \\frac{\\alpha_\\text{vis} - \\alpha_\\text{swir}}
                              {\\alpha_\\text{vis} + \\alpha_\\text{swir}}

    **Why this pair of bands works** is the whole spectral signature of snow in
    one number. In the visible, ice absorption is so weak that clean snow is
    near-perfectly reflective regardless of grain size. In the shortwave
    infrared, ice absorbs strongly and snow is dark. Nothing else common in a
    scene is bright in one and dark in the other: cloud is bright in both, rock
    and soil are dark-ish in both. The contrast is therefore a snow detector
    rather than a brightness detector, which is exactly why it survived from
    Landsat TM into the operational MODIS product.

    It is a *detector*, not a retrieval, and the reason is saturation rather
    than blindness. The index does move with grain size -- the shortwave band
    it is built from responds strongly to it -- but it moves into a corner:
    over the committed curves it runs 0.80, 0.89, 0.96, 0.98, 0.99 for radii
    of 50, 100, 250, 500 and 1000 micrometres. The first step is worth 0.10
    and the last is worth 0.007. A quantity whose sensitivity collapses by a
    factor of fifteen across the range of interest is a poor thing to invert,
    which is why grain size is retrieved from the shortwave band directly and
    not from the index built on top of it.

    References:
        Dozier, J. (1989). *Remote Sensing of Environment*, 28, 9-22.
        Hall, D. K., Riggs, G. A., & Salomonson, V. V. (1995). Development of
            methods for mapping global snow cover using MODIS data. *Remote
            Sensing of Environment*, 54(2), 127-140.
    """
    total = visible + shortwave_infrared
    if total == 0.0:
        raise ValueError("NDSI is undefined when both bands are zero.")
    return (visible - shortwave_infrared) / total
