"""What a satellite would report over this snowpack.

Takes a spectral albedo curve and answers the question an image poses: not
"what is the albedo at 560 nm" but "what number would band B3 return". The
difference is the instrument, and this module is where it enters.

Everything here is deliberately cheap. The forward model that feeds it is the
analytic one, which sweeps a few hundred wavelengths in a second, so a sweep
over grain sizes and impurity loadings is seconds rather than hours. That is
what makes the inverse direction -- given band values, recover the snow --
tractable without a surrogate, an emulator or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snow_mcrt.domain.sensor import (
    NDSI_BANDS,
    Band,
    Gaussian,
    TopHat,
    band_albedo,
    ndsi,
)


@dataclass(frozen=True)
class BandAlbedoResult:
    """Band albedos for one instrument over one snowpack."""

    instrument: str
    band_albedo: dict[str, float]
    #: ``None`` when the instrument has no NDSI pair defined.
    ndsi: float | None

    def columns(self) -> dict[str, float]:
        """Flat mapping for writing to CSV."""
        out: dict[str, float] = dict(self.band_albedo)
        if self.ndsi is not None:
            out["ndsi"] = self.ndsi
        return out


def integrate_bands(
    wavelength_nm: np.ndarray,
    albedo: np.ndarray,
    bands: dict[str, Band],
    instrument: str,
    irradiance: np.ndarray | None = None,
) -> BandAlbedoResult:
    """Integrate a spectral albedo curve over an instrument's bands.

    Args:
        wavelength_nm: Grid the albedo is given on, ascending.
        albedo: Spectral albedo on that grid.
        bands: Band name to band, e.g. ``sensor.SENTINEL2_MSI``.
        instrument: Key into :data:`~snow_mcrt.domain.sensor.NDSI_BANDS`, used
            to decide which pair the index is built from. An unknown name is
            not an error: it simply means no index is reported.
        irradiance: Incident spectral irradiance on the same grid. See
            :func:`~snow_mcrt.domain.sensor.band_albedo` for why this is
            explicit rather than assumed flat.
    """
    values = {
        name: band_albedo(wavelength_nm, albedo, band, irradiance)
        for name, band in bands.items()
    }

    index: float | None = None
    pair = NDSI_BANDS.get(instrument)
    if pair is not None and pair[0] in values and pair[1] in values:
        index = ndsi(values[pair[0]], values[pair[1]])

    return BandAlbedoResult(instrument=instrument, band_albedo=values, ndsi=index)


def response_shape_sensitivity(
    wavelength_nm: np.ndarray,
    albedo: np.ndarray,
    bands: dict[str, Band],
) -> dict[str, float]:
    """How much the *shape* of the response changes each band's answer.

    Replaces every top-hat with a Gaussian of the same full width at half
    maximum and reports the absolute difference in band albedo. Matching FWHM
    rather than area is the point: it isolates the shape from the width.

    This exists because the module ships top-hat approximations of measured
    curves, and a repository that ships an approximation owes the reader its
    size. The expected pattern is small differences in the visible, where the
    albedo curve is flat across a band, and larger ones in the near infrared,
    where ice absorption turns sharply within the band's own width.

    Returns:
        Band name to ``|alpha_gaussian - alpha_tophat|``. Bands whose response
        is not a top-hat are skipped, since there is nothing to compare.
    """
    differences: dict[str, float] = {}

    for name, band in bands.items():
        if not isinstance(band.response, TopHat):
            continue
        gaussian = Band(name, Gaussian(band.response.centre_nm, band.response.width_nm))
        differences[name] = abs(
            band_albedo(wavelength_nm, albedo, gaussian)
            - band_albedo(wavelength_nm, albedo, band)
        )

    return differences
