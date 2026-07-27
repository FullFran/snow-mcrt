"""NumPy array backend -- the reference implementation.

This is not a fallback for machines without a GPU. It is the oracle: every
physics test runs against it, because on small photon counts its output can be
compared to closed-form radiative-transfer solutions and to published
benchmark tables. If CuPy ever disagrees with NumPy, CuPy is wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyBackend:
    """Array backend on the host, using ``numpy.random.Generator``."""

    name = "numpy"

    @property
    def xp(self) -> Any:
        return np

    def rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed)

    def random_uniform(
        self, generator: np.random.Generator, shape: tuple[int, ...]
    ) -> np.ndarray:
        return generator.random(shape)

    def asarray(self, array: Any) -> np.ndarray:
        return np.asarray(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        return np.asarray(array)

    def is_available(self) -> bool:
        return True
