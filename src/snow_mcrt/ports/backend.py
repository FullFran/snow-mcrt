"""Array backend port.

The simulation core never imports NumPy or CuPy directly. It receives a
``Backend`` and works through it, so the same transport physics runs on CPU
(tests, small runs, and the reference oracle) and on GPU (production photon
counts).

Deliberately thin. NumPy and CuPy expose nearly identical array APIs, so the
port only covers what genuinely differs: random number generation, device
transfer, and module identity.

The port exists even while only the NumPy adapter is used in anger. It is what
keeps ``domain/`` free of any array-library import, which in turn is what makes
a second independent implementation possible to cross-check against.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Minimal array namespace the simulation core depends on."""

    name: str

    @property
    def xp(self) -> Any:
        """The array module itself (``numpy`` or ``cupy``)."""
        ...

    def rng(self, seed: int | None) -> Any:
        """Return a seeded random generator for this backend."""
        ...

    def random_uniform(self, generator: Any, shape: tuple[int, ...]) -> Any:
        """Uniform floats in [0, 1) allocated *on the target device*.

        Allocating on the device matters: a photon-transport loop draws random
        numbers every step for every live photon. Creating them on the host and
        copying them across is the dominant cost in a naive GPU Monte Carlo,
        and it is invisible in profiles that only time the kernel.
        """
        ...

    def asarray(self, array: Any) -> Any:
        """Move or convert an array onto this backend's device."""
        ...

    def to_numpy(self, array: Any) -> np.ndarray:
        """Copy an array back to host memory as a NumPy array."""
        ...

    def is_available(self) -> bool:
        """Whether this backend can actually run here (e.g. CUDA present)."""
        ...
