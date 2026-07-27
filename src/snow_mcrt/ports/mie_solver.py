"""Mie solver port.

Mie theory is domain physics, but the *series evaluation* is a solved numerical
problem with well-tested implementations. This port draws the line between the
two: ``domain/`` owns what the efficiencies mean and how they combine into
snowpack optical properties, while the recursion over Riccati-Bessel functions
lives behind an adapter.

The same reasoning as the array backend applies. A port here means a second
independent solver can be dropped in and cross-checked, which is the only way
to tell a subtle convergence failure at large size parameters from a real
physical result.

Sign convention (enforced at this boundary): the complex refractive index is
passed as ``m = n + ik`` with ``k >= 0`` for an absorbing medium. Adapters are
responsible for converting to whatever their library expects -- miepython, for
one, uses ``m = n - ik``. Getting this backwards yields gain instead of
absorption and the error is easy to miss because the magnitudes still look
plausible.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MieSolver(Protocol):
    """Evaluates Mie efficiencies for homogeneous spheres."""

    name: str

    def efficiencies(self, m: Any, x: Any) -> tuple[Any, Any, Any]:
        """Return ``(q_ext, q_sca, g)`` for relative index ``m``, size ``x``.

        Args:
            m: Complex relative refractive index ``n + ik``, ``k >= 0``.
                Scalar or array.
            x: Size parameter ``2*pi*r*n_medium/lambda``. Scalar or array,
                broadcastable against ``m``.

        Returns:
            Extinction efficiency, scattering efficiency, and asymmetry
            parameter ``g = <cos(theta)>``, broadcast to a common shape.
        """
        ...
