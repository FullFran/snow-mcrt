"""Complex refractive index and bulk absorption.

Unit convention for the whole package, fixed here because mixing the two is the
classic silent error in this field:

- **Wavelengths are nanometres** at every public boundary. The snow-optics
  literature quotes them that way and so do the tabulated datasets.
- **Lengths are metres** everywhere else: grain radii, absorption
  coefficients, optical depths, snowpack geometry.

Sign convention: the complex index is ``m = n + ik`` with ``k >= 0`` for an
absorbing medium. This is the Bohren & Huffman convention and it is what the
domain speaks. Libraries that use ``m = n - ik`` are converted at their adapter
boundary, never here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NM_PER_M = 1e9


@dataclass(frozen=True)
class OpticalConstants:
    """Tabulated complex refractive index on a wavelength grid.

    Args:
        wavelength_nm: Strictly ascending wavelengths in nanometres.
        n: Real part of the refractive index. Must be positive.
        k: Imaginary part (absorptive). Must be non-negative; ``k = 0`` is a
            transparent medium and is legitimate.
        name: Human-readable label, carried into manifests so a result can be
            traced back to the dataset that produced it.
    """

    wavelength_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray
    name: str = "unnamed"

    def __post_init__(self) -> None:
        lam, n, k = self.wavelength_nm, self.n, self.k
        if not (lam.shape == n.shape == k.shape):
            raise ValueError(
                f"wavelength/n/k shapes disagree: "
                f"{lam.shape}, {n.shape}, {k.shape}"
            )
        if lam.ndim != 1:
            raise ValueError(f"expected a 1-D wavelength grid, got {lam.ndim}-D")
        if lam.size < 2:
            raise ValueError("need at least two wavelengths to interpolate")
        if not np.all(np.diff(lam) > 0):
            raise ValueError("wavelengths must be strictly ascending")
        if np.any(n <= 0):
            raise ValueError("refractive index n must be positive")
        if np.any(k < 0):
            raise ValueError(
                "k must be non-negative under the m = n + ik convention; "
                "a negative k describes gain, not absorption"
            )

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        """Inclusive bounds of the tabulated grid."""
        return float(self.wavelength_nm[0]), float(self.wavelength_nm[-1])

    def m_at(self, wavelength_nm: np.ndarray | float) -> np.ndarray:
        """Interpolate the complex index ``n + ik`` at the given wavelengths.

        Extrapolation is refused rather than silently clamped. Ice absorption
        varies over ten orders of magnitude across the solar spectrum, so a
        clamped endpoint is not a small error -- it is a different physical
        material.

        Raises:
            ValueError: If any requested wavelength lies outside the table.
        """
        lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        lo, hi = self.wavelength_range_nm
        if np.any(lam < lo) or np.any(lam > hi):
            outside = lam[(lam < lo) | (lam > hi)]
            raise ValueError(
                f"wavelengths outside the tabulated range [{lo}, {hi}] nm: "
                f"{outside[:5]}{'...' if outside.size > 5 else ''}"
            )
        n = np.interp(lam, self.wavelength_nm, self.n)
        k = np.interp(lam, self.wavelength_nm, self.k)
        return n + 1j * k

    def absorption_coefficient(
        self, wavelength_nm: np.ndarray | float
    ) -> np.ndarray:
        """Bulk absorption coefficient in inverse metres.

        The Beer-Lambert coefficient of the *material itself*, before any
        scattering geometry enters:

        .. math::
            \\gamma = \\frac{4 \\pi k}{\\lambda}

        Intensity in a non-scattering slab falls as ``exp(-gamma * z)``.
        """
        lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        k = np.imag(self.m_at(lam))
        return 4.0 * np.pi * k / (lam / NM_PER_M)


def size_parameter(
    radius_m: np.ndarray | float,
    wavelength_nm: np.ndarray | float,
    n_medium: float = 1.0,
) -> np.ndarray:
    """Mie size parameter ``x = 2*pi*r*n_medium/lambda``.

    Args:
        radius_m: Sphere radius in metres.
        wavelength_nm: Vacuum wavelength in nanometres.
        n_medium: Real refractive index of the surrounding medium. For snow the
            grains sit in air, so the default of 1.0 is the physical case; the
            argument exists because the same Mie machinery is used for
            particles embedded in ice or water.

    Returns:
        Size parameter, broadcast over the inputs. Always at least 1-D, so
        that scalar and array calls can be handled by the same downstream
        code -- matching :meth:`OpticalConstants.m_at`.
    """
    r = np.atleast_1d(np.asarray(radius_m, dtype=float))
    lam_m = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) / NM_PER_M
    if np.any(r < 0):
        raise ValueError("radius must be non-negative")
    if np.any(lam_m <= 0):
        raise ValueError("wavelength must be positive")
    return 2.0 * np.pi * r * n_medium / lam_m
