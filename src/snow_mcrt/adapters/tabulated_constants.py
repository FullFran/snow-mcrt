"""Optical constants loaded from a whitespace-delimited table.

The format is the lowest common denominator of the published datasets: three
columns -- wavelength, ``n``, ``k`` -- with ``#`` comments. Warren & Brandt
(2008) and the refractiveindex.info exports both reduce to it with no parsing
cleverness, which is the point.

Every loaded table carries its ``name`` into the run manifest. A spectral
albedo curve is only reproducible if the dataset behind it is identified, and
"the ice constants" is not an identification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from snow_mcrt.domain.optics import OpticalConstants


class TabulatedConstants:
    """Reads ``wavelength n k`` rows from a text file.

    Args:
        path: File to read.
        name: Dataset label for manifests. Defaults to the file stem.
        wavelength_scale_to_nm: Multiplier converting the file's wavelength
            column to nanometres. Published tables are quoted in micrometres
            as often as nanometres, and the mistake is invisible in a plot
            whose axis was labelled from the same wrong assumption.
    """

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        wavelength_scale_to_nm: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.wavelength_scale_to_nm = float(wavelength_scale_to_nm)

    def load(self) -> OpticalConstants:
        """Parse the table.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If the table does not have three columns, or the
                values violate the constraints of ``OpticalConstants``.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"optical constants file not found: {self.path}")

        table = np.loadtxt(self.path, comments="#", ndmin=2)
        if table.shape[1] != 3:
            raise ValueError(
                f"{self.path}: expected 3 columns (wavelength, n, k), "
                f"got {table.shape[1]}"
            )

        lam = table[:, 0] * self.wavelength_scale_to_nm
        order = np.argsort(lam)
        return OpticalConstants(
            wavelength_nm=lam[order],
            n=table[order, 1],
            k=table[order, 2],
            name=self.name,
        )
