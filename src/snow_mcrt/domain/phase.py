"""Scattering phase functions and how to sample them.

Two representations, and they are not interchangeable:

- **Henyey-Greenstein** -- a one-parameter analytic form fitted to reproduce
  the asymmetry parameter ``g``. It has a closed-form inverse CDF, so sampling
  costs one uniform draw and no table lookup.
- **Tabulated** -- an arbitrary ``p(mu)`` on a grid, sampled by inverting an
  interpolated CDF. This is how the full Mie phase function enters.

Both normalise as ``integral p(mu) dmu = 1`` over ``mu`` in ``[-1, 1]``, which
is the convention the transport loop assumes. Phase functions in the
literature are as often normalised over ``4*pi`` steradians, differing by a
factor of ``2*pi``; conversion happens at the adapter boundary.

Everything here is written in plain arithmetic against a backend's array
module, so the same code samples on host and device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from snow_mcrt.ports.backend import Backend

# Below this, the closed-form HG inverse CDF loses precision to cancellation
# and the isotropic branch is both exact and better conditioned.
_ISOTROPIC_G_THRESHOLD = 1e-6


def henyey_greenstein_pdf(mu: Any, g: float) -> Any:
    """Henyey-Greenstein phase function, normalised over ``mu`` in ``[-1, 1]``.

    .. math::
        p(\\mu) = \\frac{1 - g^2}{2 (1 + g^2 - 2 g \\mu)^{3/2}}

    Args:
        mu: Cosine of the scattering angle.
        g: Asymmetry parameter in ``(-1, 1)``.
    """
    if not -1.0 < g < 1.0:
        raise ValueError(f"asymmetry parameter must lie in (-1, 1), got {g}")
    return (1.0 - g**2) / (2.0 * (1.0 + g**2 - 2.0 * g * mu) ** 1.5)


def sample_henyey_greenstein(
    backend: Backend, generator: Any, g: float, shape: tuple[int, ...]
) -> Any:
    """Draw scattering-angle cosines from the Henyey-Greenstein distribution.

    Inverts the CDF in closed form, so each sample costs one uniform draw:

    .. math::
        \\mu = \\frac{1}{2g}\\left[1 + g^2 -
               \\left(\\frac{1 - g^2}{1 - g + 2 g u}\\right)^2\\right]

    Args:
        backend: Array backend; the uniforms are allocated on its device.
        generator: Seeded generator from ``backend.rng``.
        g: Asymmetry parameter in ``(-1, 1)``.
        shape: Output shape.

    Returns:
        Cosines in ``[-1, 1]``, on the backend's device.
    """
    if not -1.0 < g < 1.0:
        raise ValueError(f"asymmetry parameter must lie in (-1, 1), got {g}")

    u = backend.random_uniform(generator, shape)
    if abs(g) < _ISOTROPIC_G_THRESHOLD:
        return 2.0 * u - 1.0

    term = (1.0 - g**2) / (1.0 - g + 2.0 * g * u)
    mu = (1.0 + g**2 - term**2) / (2.0 * g)
    # Mathematically bounded already; this only absorbs float error at the
    # endpoints, where a mu of 1 + 1e-16 would produce a NaN direction.
    return backend.xp.clip(mu, -1.0, 1.0)


def forward_peaked_mu_grid(
    n_points: int = 4000, min_angle_rad: float = 1e-7
) -> np.ndarray:
    """Angular grid that resolves the forward diffraction peak.

    Returns ascending ``mu = cos(theta)`` with ``theta`` spaced logarithmically
    from ``min_angle_rad`` to ``pi``, plus exact forward scattering.

    **This grid is not a refinement, it is a correctness requirement.** A large
    sphere concentrates most of its scattered energy into a diffraction peak of
    angular width of order ``1/x``. For a 100 um snow grain in the visible,
    ``x ~ 1.3e3``, so the peak spans microradians -- and in ``mu`` it is
    narrower still, since ``1 - mu ~ theta^2 / 2`` puts it within ``1e-7`` of
    unity. A uniformly spaced ``mu`` grid steps straight over it: the tabulated
    function then integrates to roughly twenty instead of one, and its mean
    cosine comes out above one, which is not a physical quantity at all.

    Args:
        n_points: Number of logarithmically spaced angles.
        min_angle_rad: Smallest non-zero angle. Should sit well below ``1/x``
            for the largest size parameter in play.
    """
    if n_points < 2:
        raise ValueError("need at least two grid points")
    if not 0 < min_angle_rad < np.pi:
        raise ValueError("min_angle_rad must lie in (0, pi)")
    theta = np.concatenate(
        ([0.0], np.logspace(np.log10(min_angle_rad), np.log10(np.pi), n_points))
    )
    # Below about 1e-7 rad, neighbouring angles map to the same float64
    # cosine: 1 - cos(theta) ~ theta^2 / 2 falls under the 1e-16 spacing at
    # unity. Those collapsed points are dropped rather than left to break
    # strict monotonicity downstream. It is a hard precision floor, not a
    # tuning choice -- resolving finer would require carrying 1 - mu instead
    # of mu, which the rest of the pipeline does not need.
    return np.unique(np.cos(theta))


@dataclass(frozen=True)
class TabulatedPhaseFunction:
    """A phase function sampled from a tabulated ``p(mu)``.

    Args:
        mu: Strictly ascending cosines spanning ``[-1, 1]``.
        p: Phase function values, normalised so ``integral p dmu = 1``.
        name: Label carried into run manifests.
    """

    mu: np.ndarray
    p: np.ndarray
    name: str = "tabulated"

    def __post_init__(self) -> None:
        if self.mu.shape != self.p.shape:
            raise ValueError(
                f"mu and p shapes disagree: {self.mu.shape}, {self.p.shape}"
            )
        if self.mu.ndim != 1 or self.mu.size < 2:
            raise ValueError("expected a 1-D grid with at least two points")
        if not np.all(np.diff(self.mu) > 0):
            raise ValueError("mu must be strictly ascending")
        if np.any(self.p < 0):
            raise ValueError("a phase function cannot be negative")
        norm = float(np.trapezoid(self.p, self.mu))
        if not np.isclose(norm, 1.0, rtol=2e-2):
            raise ValueError(
                f"phase function integrates to {norm:.4f}, not 1. Either the "
                f"normalisation convention is wrong (a factor of 2*pi is the "
                f"usual culprit) or the grid fails to resolve the forward "
                f"peak -- see forward_peaked_mu_grid"
            )

    @property
    def mean_cosine(self) -> float:
        """Asymmetry parameter implied by the table.

        Must agree with the ``g`` returned by the Mie solver. The two are
        computed by completely different routes, so a disagreement means the
        angular grid is too coarse.
        """
        return float(np.trapezoid(self.p * self.mu, self.mu))

    def cumulative(self) -> np.ndarray:
        """Cumulative distribution on the tabulated grid, rising 0 to 1."""
        from scipy.integrate import cumulative_trapezoid

        cdf = cumulative_trapezoid(self.p, self.mu, initial=0.0)
        return cdf / cdf[-1]

    def sample(
        self, backend: Backend, generator: Any, shape: tuple[int, ...]
    ) -> Any:
        """Draw scattering-angle cosines by inverting the interpolated CDF."""
        xp = backend.xp
        cdf = self.cumulative()
        # A CDF is only invertible where it strictly increases; the deep
        # backscatter tail of a large sphere is flat to floating point.
        keep = np.concatenate(([True], np.diff(cdf) > 0))
        cdf_x = backend.asarray(cdf[keep])
        mu_x = backend.asarray(self.mu[keep])
        u = backend.random_uniform(generator, shape)
        return xp.interp(u, cdf_x, mu_x)


def tabulate_mie_phase_function(
    solver: Any,
    m: complex,
    x: float,
    mu: np.ndarray | None = None,
    name: str | None = None,
) -> TabulatedPhaseFunction:
    """Build a tabulated phase function from a Mie solver.

    Args:
        solver: A :class:`~snow_mcrt.ports.mie_solver.MieSolver` that also
            implements ``phase_function``.
        m: Complex relative index ``n + ik``, ``k >= 0``.
        x: Size parameter.
        mu: Angular grid. Defaults to :func:`forward_peaked_mu_grid`, which is
            what makes the result correct at snow size parameters.
        name: Label for manifests.

    Returns:
        The tabulated phase function, validated for normalisation on
        construction.
    """
    grid = forward_peaked_mu_grid() if mu is None else np.asarray(mu, dtype=float)
    p = solver.phase_function(m, x, grid)
    return TabulatedPhaseFunction(
        mu=grid, p=np.asarray(p, dtype=float), name=name or f"mie(x={x:g})"
    )
