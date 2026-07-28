"""Mie solver backed by the ``miepython`` package.

``miepython`` implements the Bohren & Huffman recursion with the downward
recurrence for the logarithmic derivative, which is what keeps it stable at the
large size parameters snow needs (a 1 mm grain at 400 nm is ``x ~ 1.6e4``).

**Sign convention.** ``miepython`` takes ``m = n - ik``. The domain speaks
``m = n + ik``. The conjugation happens here, at the boundary, and nowhere
else. Passing the domain convention straight through does not raise -- it
quietly models a medium with gain, and the resulting albedo curves look
entirely reasonable until they are compared against a benchmark.
"""

from __future__ import annotations

import miepython
import numpy as np


class MiepythonSolver:
    """Adapter over ``miepython.efficiencies_mx``."""

    name = "miepython"
    version = getattr(miepython, "__version__", "unknown")

    def efficiencies(
        self, m: np.ndarray | complex, x: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q_ext, q_sca, g)`` under the ``m = n + ik`` convention."""
        m_arr = np.asarray(m, dtype=complex)
        x_arr = np.asarray(x, dtype=float)
        if np.any(m_arr.imag < 0):
            raise ValueError(
                "the domain convention is m = n + ik with k >= 0; a negative "
                "imaginary part describes gain, not absorption"
            )
        if np.any(x_arr < 0):
            raise ValueError("size parameter must be non-negative")

        # miepython indexes its inputs with a scalar loop, so it accepts only
        # 0-d or 1-D arrays: a 0-d input raises on len(), and a 2-D one raises
        # on an ambiguous truth value. Flatten to 1-D here and restore the
        # caller's shape afterwards, so the domain can hand over any broadcast
        # grid it likes -- a (radius, wavelength) quadrature, for one.
        m_arr, x_arr = np.broadcast_arrays(np.atleast_1d(m_arr), np.atleast_1d(x_arr))
        shape = m_arr.shape
        # miepython's convention: m = n - ik.
        q_ext, q_sca, _q_back, g = miepython.efficiencies_mx(
            np.conjugate(m_arr).ravel(), x_arr.ravel()
        )
        return (
            np.asarray(q_ext, dtype=float).reshape(shape),
            np.asarray(q_sca, dtype=float).reshape(shape),
            np.asarray(g, dtype=float).reshape(shape),
        )

    def phase_function(
        self, m: complex, x: float, mu: np.ndarray
    ) -> np.ndarray:
        """Return ``p(mu)`` normalised so ``integral p dmu = 1``.

        ``miepython``'s ``norm="one"`` normalises over 4*pi steradians, so the
        factor of ``2*pi`` converts to the per-``mu`` convention the domain
        uses.
        """
        if np.imag(m) < 0:
            raise ValueError(
                "the domain convention is m = n + ik with k >= 0; a negative "
                "imaginary part describes gain, not absorption"
            )
        mu_arr = np.asarray(mu, dtype=float)
        if np.any(np.abs(mu_arr) > 1.0):
            raise ValueError("mu must be a cosine, within [-1, 1]")
        intensity = miepython.i_unpolarized(
            np.conjugate(np.complex128(m)), float(x), mu_arr, norm="one"
        )
        return 2.0 * np.pi * np.asarray(intensity, dtype=float)
