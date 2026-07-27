"""CuPy array backend -- production photon counts.

Importing this module never fails on a machine without CUDA. ``is_available()``
answers that question, so a caller can select a backend without guarding the
import, and the test suite can assert the contract holds on CPU-only hardware.

Where the GPU actually pays: the transport loop, where millions of live photons
advance together. Mie evaluation over a wavelength grid is thousands of values
and belongs on the host -- running it here would be motion, not speed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - import success depends on the host
    import cupy as _cupy
except ImportError:  # pragma: no cover
    _cupy = None


class CupyBackend:
    """Array backend on a CUDA device.

    Raises:
        RuntimeError: On construction, if CuPy is not importable. Failing here
            rather than at first array use keeps the error at the point where
            the backend was chosen, which is where the fix belongs.
    """

    name = "cupy"

    def __init__(self) -> None:
        if _cupy is None:
            raise RuntimeError(
                "CuPy is not installed. Install the extra matching your CUDA "
                "version, e.g. `pip install snow-mcrt[gpu]`, or use NumpyBackend."
            )

    @property
    def xp(self) -> Any:
        return _cupy

    def rng(self, seed: int | None) -> Any:
        return _cupy.random.default_rng(seed)

    def random_uniform(self, generator: Any, shape: tuple[int, ...]) -> Any:
        # Allocated on the device by construction. Drawing on the host and
        # copying across is the dominant cost in a naive GPU Monte Carlo, and
        # it does not show up in kernel timings.
        return generator.random(shape)

    def asarray(self, array: Any) -> Any:
        return _cupy.asarray(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        return _cupy.asnumpy(array)

    def is_available(self) -> bool:
        if _cupy is None:
            return False
        try:
            return _cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


def cupy_is_importable() -> bool:
    """Whether CuPy could be imported at all, without constructing a backend."""
    return _cupy is not None
