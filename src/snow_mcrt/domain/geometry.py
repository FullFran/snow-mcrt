"""Where a photon meets something that is not snow.

A buried object is a region with its own refractive index, extinction and
single-scattering albedo. Everything the transport loop needs from its shape is
one question, asked of every live photon at every step: *how far to your next
interface, and which way does it face there?*

That question is the whole of this module. Keeping it here rather than inside
the transport loop is what will let a second shape be added without touching
any physics.

**Axis-aligned boxes, and not more.** A rectangular block buried flat is the
geometry the detectability note describes, and the slab method solves it
exactly with six divisions and no branching -- it vectorises with nothing but
element-wise minima. A general polyhedron needs a bilinear patch intersection
per face, which is an order of magnitude more arithmetic for a shape nobody has
asked for yet. When one is asked for, it goes behind the same two-value
interface this returns.

Conventions, shared with :mod:`~snow_mcrt.domain.fresnel`:

- ``z`` is depth, positive downward.
- Positions and directions are ``(n, 3)``; directions are unit vectors.
- Returned normals point *against* the incoming ray, which is what Fresnel
  expects, so the caller never has to work out which side it hit.
- A photon that will not meet the surface gets ``inf``. The transport loop
  compares that against a sampled free path, and infinity always loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Rays exactly parallel to a slab divide by zero. Substituting a tiny
# denominator sends the crossing to +-inf, which is the geometrically correct
# answer -- such a ray never enters that slab -- and avoids a nan that would
# then have to be filtered out of every comparison downstream.
_PARALLEL_EPSILON = 1e-300

# A photon sitting exactly on a face must not immediately re-hit it. One
# nanometre is far below any optical length in snow -- the transport mean free
# path is millimetres -- and far above the rounding of a position accumulated
# over a few thousand steps.
_SURFACE_EPSILON = 1e-9


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangular region of a different medium.

    Args:
        lower: ``(x, y, z)`` of the corner with the smallest coordinates, m.
        upper: ``(x, y, z)`` of the opposite corner, m.

    The depth convention makes ``lower[2]`` the top face of a buried object,
    which reads backwards until you remember that ``z`` grows downward.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("a box is defined by two points in three dimensions")
        if np.any(upper <= lower):
            raise ValueError(
                "every upper coordinate must exceed its lower one; a box with "
                "no volume has no inside for a photon to be in"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def centre(self) -> np.ndarray:
        """Centre of the box, m."""
        return 0.5 * (self.lower + self.upper)

    @property
    def size(self) -> np.ndarray:
        """Edge lengths, m."""
        return self.upper - self.lower

    @property
    def top_depth_m(self) -> float:
        """Depth of the shallowest face -- the burial depth."""
        return float(self.lower[2])

    def contains(self, position: Any, xp: Any = np) -> Any:
        """Whether each position lies inside the box.

        The boundary counts as inside. A photon that has just refracted
        through a face is sitting exactly on it, and calling that outside
        would send it straight back out.
        """
        lower = xp.asarray(self.lower)
        upper = xp.asarray(self.upper)
        return xp.all((position >= lower) & (position <= upper), axis=-1)

    def distance_to_surface(
        self, position: Any, direction: Any, xp: Any = np
    ) -> tuple[Any, Any]:
        """Distance to this box's surface, and the normal there.

        The slab method: intersect the ray with each pair of parallel planes
        and keep the overlap of the three intervals. Works from inside and
        outside without a special case -- from inside, the near crossing is
        behind the photon and the far one is the exit.

        Args:
            position: ``(n, 3)`` positions, m.
            direction: ``(n, 3)`` unit directions.
            xp: Array namespace.

        Returns:
            ``(distance, normal)``. ``distance`` is ``inf`` where the ray
            misses or points away. ``normal`` is a unit vector oriented
            against the ray; it is meaningless where the distance is
            infinite, and the caller must not use it there.
        """
        lower = xp.asarray(self.lower)
        upper = xp.asarray(self.upper)

        # Signed so that a negative component still yields a valid interval
        # once the two crossings are sorted below.
        safe = xp.where(
            xp.abs(direction) < _PARALLEL_EPSILON,
            _PARALLEL_EPSILON,
            direction,
        )
        t_lower = (lower - position) / safe
        t_upper = (upper - position) / safe
        t_near = xp.minimum(t_lower, t_upper)
        t_far = xp.maximum(t_lower, t_upper)

        entry = xp.max(t_near, axis=-1)
        exit_ = xp.min(t_far, axis=-1)

        # A hit needs the three intervals to overlap, and the overlap to lie
        # ahead of the photon. Entry beyond the epsilon means the photon is
        # outside and approaching; otherwise it is already inside and the
        # relevant crossing is the exit.
        hits = exit_ > xp.maximum(entry, _SURFACE_EPSILON)
        from_outside = entry > _SURFACE_EPSILON
        distance = xp.where(from_outside, entry, exit_)
        distance = xp.where(hits, distance, xp.inf)

        # Which slab was decisive. On entry it is the axis whose near crossing
        # is latest; on exit, the axis whose far crossing is earliest.
        axis = xp.where(
            from_outside,
            xp.argmax(t_near, axis=-1),
            xp.argmin(t_far, axis=-1),
        )
        eye = xp.eye(3)
        normal = eye[axis]
        # Orient against the ray, so Fresnel sees a non-negative cosine
        # whichever face and whichever side this is.
        facing = xp.sum(normal * direction, axis=-1, keepdims=True)
        normal = xp.where(facing > 0, -normal, normal)
        return distance, normal
