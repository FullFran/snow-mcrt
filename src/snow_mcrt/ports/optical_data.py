"""Optical constants port.

Where the refractive index of ice (or of an impurity) comes from is an
infrastructure concern: a tabulated file, a published fit, a synthetic dataset
built for a test. What the domain needs is a wavelength grid and the complex
index on it.

Keeping this behind a port is what lets the physics tests run on an analytic
dataset with known answers instead of on a multi-megabyte data file whose
correctness is itself an open question.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from snow_mcrt.domain.optics import OpticalConstants


@runtime_checkable
class OpticalConstantsSource(Protocol):
    """Supplies tabulated complex refractive index over wavelength."""

    name: str

    def load(self) -> OpticalConstants:
        """Return the tabulated constants, wavelengths ascending."""
        ...
