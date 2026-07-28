"""Three-dimensional photon transport with a real surface.

The plane-parallel engine in :mod:`~snow_mcrt.domain.transport` answers one
question extremely well: what fraction of light comes back. It cannot answer
*where* it comes back, because it carries no lateral position -- a photon is a
depth and a direction cosine and nothing else. Every spatially resolved
observable the detectability work needs -- the sensitivity banana, a
source-detector separation, an object at a position -- is invisible to it.

This module carries the full state: position ``(x, y, z)`` and direction
``(u_x, u_y, u_z)``, with ``z`` the depth, positive downward. Photons remain
an axis of the array, exactly as before; the arrays are simply wider.

Two things follow that the 1-D engine never had to face.

**The surface is an interface, not an exit.** Snow is ``n = 1.31``, and 58% of
the diffuse flux arriving at the boundary from inside arrives beyond the
critical angle and is reflected *entirely*. See
:mod:`~snow_mcrt.domain.fresnel`. Ignoring it does not produce a slightly wrong
answer; it produces a different problem.

**A free path competes with the distance to the interface.** A photon may not
simply step by its sampled free path: if that step would carry it through the
surface, it stops at the surface instead, and Fresnel decides what happens
next. Stepping first and testing afterwards is the classic way to walk a
photon through a wall, and the resulting profile looks plausible.

The step order is the standard one for heterogeneous geometry, as in Wang,
Jacques & Zheng (1995): sample the free path, compare it against the distance
to the nearest interface, and take whichever comes first. That structure is
what will let a buried object with its own index and extinction be added
without disturbing anything here.

Escaping weight is accumulated into radial bins, which is the whole point:
``reflectance`` is the radially resolved diffuse reflectance ``R(rho)`` that
:meth:`~snow_mcrt.domain.diffusion.DiffusionParameters.diffuse_reflectance`
predicts in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from snow_mcrt.domain.analytic import delta_eddington_scaling
from snow_mcrt.domain.fresnel import fresnel_reflectance, refract, specular_reflect
from snow_mcrt.domain.geometry import Box
from snow_mcrt.domain.transport import (
    _ISOTROPIC_G_THRESHOLD,
    TransportConfig,
    _sample_hg_cosines,
)
from snow_mcrt.ports.backend import Backend

# Below this the scattering rotation's sqrt(1 - u_z^2) loses its digits and the
# degenerate branch is used instead. Not a tunable: it is where the general
# formula stops being computable, not where it stops being accurate.
_POLE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BuriedObject:
    """A region of different medium inside the snowpack.

    Args:
        box: Where it is.
        extinction_coefficient: ``beta`` inside it, m^-1. **Zero is allowed**
            and is the most interesting case: a void. A cavity scatters
            nothing and absorbs nothing, so photons cross it in a straight
            line, and the perturbation to the diffuse field is larger than an
            absorber of the same size produces -- see ``docs/detectability.md``.
        single_scattering_albedo: ``omega`` inside it.
        asymmetry: ``g`` inside it.
        refractive_index: Its index. Different from the snow's means a Fresnel
            interface at every face, which for a void is the whole signal.
    """

    box: Box
    extinction_coefficient: float = 0.0
    single_scattering_albedo: float = 0.0
    asymmetry: float = 0.0
    refractive_index: float = 1.0

    def __post_init__(self) -> None:
        if self.extinction_coefficient < 0:
            raise ValueError("extinction coefficient must be non-negative")
        if not 0 <= self.single_scattering_albedo <= 1:
            raise ValueError("single-scattering albedo must lie in [0, 1]")
        if not -1 < self.asymmetry < 1:
            raise ValueError("asymmetry parameter must lie in (-1, 1)")
        if self.refractive_index < 1.0:
            raise ValueError("a refractive index below one is unphysical here")


def _sample_hg_cosines_array(
    backend: Backend, generator: Any, g: Any, n: int
) -> Any:
    """Henyey-Greenstein cosines with a per-photon asymmetry.

    The scalar sampler in :mod:`~snow_mcrt.domain.transport` cannot serve two
    media at once. Same inverse CDF, with the isotropic limit selected per
    element rather than by an ``if`` -- and, as everywhere in this engine, the
    general branch is evaluated even where it is not selected, so its
    denominator is clamped rather than merely avoided.
    """
    xp = backend.xp
    u = backend.random_uniform(generator, (n,))
    safe_g = xp.where(xp.abs(g) < _ISOTROPIC_G_THRESHOLD, 1.0, g)
    term = (1.0 - safe_g**2) / (1.0 - safe_g + 2.0 * safe_g * u)
    anisotropic = xp.clip(
        (1.0 + safe_g**2 - term**2) / (2.0 * safe_g), -1.0, 1.0
    )
    return xp.where(xp.abs(g) < _ISOTROPIC_G_THRESHOLD, 2.0 * u - 1.0, anisotropic)


def log_radial_edges(inner_m: float, outer_m: float, count: int) -> np.ndarray:
    """Logarithmically spaced bin edges for a radial profile.

    Logarithmic because ``R(rho)`` falls by orders of magnitude over the range
    that matters. Uniform bins would spend almost every bin on the tail, where
    there is nothing, and leave the near field -- where the profile actually
    has structure -- in one or two.

    Args:
        inner_m: Innermost edge, metres. Must be positive; the origin itself
            is a singularity of the diffusion solution this is compared to.
        outer_m: Outermost edge, metres.
        count: Number of bins, so ``count + 1`` edges.
    """
    if inner_m <= 0:
        raise ValueError("the inner edge must be positive; R(0) is singular")
    if outer_m <= inner_m:
        raise ValueError("the outer edge must exceed the inner one")
    if count < 1:
        raise ValueError("need at least one bin")
    return np.logspace(np.log10(inner_m), np.log10(outer_m), count + 1)


@dataclass(frozen=True)
class Transport3DResult:
    """Where the energy went, and where it came back out.

    Args:
        reflected: Fraction escaping through the surface.
        absorbed: Fraction deposited in the medium.
        truncated: Fraction still in flight when the step budget ran out.
        binned_weight: Escaped fraction per radial bin.
        outside_bins: Escaped fraction falling outside the binned range.
            Reported rather than dropped, so a profile can never look complete
            when it is not.
        edges_m: Bin edges.
        scatters: Steps taken.
        final_directions: Directions of photons still alive at the end, or
            ``None`` unless requested.
    """

    reflected: float
    absorbed: float
    truncated: float
    binned_weight: np.ndarray
    outside_bins: float
    edges_m: np.ndarray
    scatters: int
    config: TransportConfig
    surface_index: float
    final_directions: np.ndarray | None = None

    @property
    def energy_balance(self) -> float:
        """``reflected + absorbed + truncated - 1``. Zero, or a bug."""
        return self.reflected + self.absorbed + self.truncated - 1.0

    @property
    def bin_centres_m(self) -> np.ndarray:
        """Geometric centres, matching the logarithmic spacing."""
        return np.sqrt(self.edges_m[:-1] * self.edges_m[1:])

    @property
    def annulus_area_m2(self) -> np.ndarray:
        """Area of each annulus, m^2."""
        return np.pi * (self.edges_m[1:] ** 2 - self.edges_m[:-1] ** 2)

    @property
    def reflectance(self) -> np.ndarray:
        """``R(rho)``: escaped fraction per unit area, m^-2.

        This is the quantity diffusion theory predicts in closed form, and the
        one a source-detector pair measures.
        """
        return self.binned_weight / self.annulus_area_m2


def _initial_directions(
    backend: Backend, generator: Any, n: int, incidence: str | float
) -> Any:
    """Starting directions, all entering downward.

    ``"collimated"`` is a pencil beam straight down -- the source diffusion
    theory's Green's function assumes, and the one a fibre or a laser spot
    approximates. ``"diffuse"`` is Lambertian, matching the plane-parallel
    engine and its analytic oracle, which is what makes the two comparable.
    """
    xp = backend.xp
    if incidence == "collimated":
        mu = xp.ones(n)
    elif incidence == "diffuse":
        # cos-weighted: mu = sqrt(u) gives a Lambertian hemisphere.
        mu = xp.sqrt(backend.random_uniform(generator, (n,)))
    elif isinstance(incidence, (int, float)):
        mu = xp.full(n, float(incidence))
    else:
        raise ValueError(
            f"unknown incidence {incidence!r}; expected 'collimated', "
            f"'diffuse', or a direction cosine"
        )
    sin_theta = xp.sqrt(xp.maximum(1.0 - mu**2, 0.0))
    phi = 2.0 * np.pi * backend.random_uniform(generator, (n,))
    return xp.stack(
        [sin_theta * xp.cos(phi), sin_theta * xp.sin(phi), mu], axis=-1
    )


def _fresnel_pairwise(cos_incident: Any, n_from: Any, n_to: Any, xp: Any) -> Any:
    """Fresnel where the two indices vary per photon.

    :func:`~snow_mcrt.domain.fresnel.fresnel_reflectance` takes scalar indices
    because the snow surface has only one pair. A face of a buried object is
    crossed in both directions at once by different photons, so the pair is an
    array. Same expression, same clamping inside the total-reflection region
    for the same reason: the unselected branch is still evaluated.
    """
    cos_i = xp.clip(cos_incident, 0.0, 1.0)
    ratio = n_from / n_to
    sin_t_sq = ratio**2 * (1.0 - cos_i**2)
    total = sin_t_sq >= 1.0
    cos_t = xp.where(total, 1.0, xp.sqrt(xp.maximum(1.0 - sin_t_sq, 0.0)))
    cos_i = xp.where(total, 1.0, cos_i)
    r_s = (n_from * cos_i - n_to * cos_t) / (n_from * cos_i + n_to * cos_t)
    r_p = (n_from * cos_t - n_to * cos_i) / (n_from * cos_t + n_to * cos_i)
    matched = n_from == n_to
    reflectance = xp.where(total, 1.0, 0.5 * (r_s**2 + r_p**2))
    return xp.where(matched, 0.0, reflectance)


def _refract_pairwise(
    direction: Any, normal: Any, n_from: Any, n_to: Any, xp: Any
) -> Any:
    """Snell's law with per-photon indices."""
    ratio = (n_from / n_to)[..., None]
    cos_i = -xp.sum(direction * normal, axis=-1, keepdims=True)
    cos_t = xp.sqrt(xp.maximum(1.0 - ratio**2 * (1.0 - cos_i**2), 0.0))
    return ratio * direction + (ratio * cos_i - cos_t) * normal


def _rotate(backend: Backend, direction: Any, cos_theta: Any, phi: Any) -> Any:
    """Deflect each direction by ``theta`` about itself, then ``phi`` around it.

    The standard MCML rotation. Its denominator ``sqrt(1 - u_z^2)`` vanishes
    when the photon travels along the axis, which is not a rare event: a
    collimated beam starts there, so it happens to every photon on step one.

    The guard has to do two things. It has to select the degenerate branch,
    and it has to keep the general branch from being *evaluated* into a nan --
    ``xp.where`` computes both sides and a nan in the discarded one still
    propagates through the multiply. Hence the clamped denominator.
    """
    xp = backend.xp
    ux, uy, uz = direction[..., 0], direction[..., 1], direction[..., 2]
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta**2, 0.0))
    cos_phi, sin_phi = xp.cos(phi), xp.sin(phi)

    denom = xp.sqrt(xp.maximum(1.0 - uz**2, _POLE_TOLERANCE))
    general = xp.stack(
        [
            sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta,
            sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta,
            -denom * sin_theta * cos_phi + uz * cos_theta,
        ],
        axis=-1,
    )
    # At the pole the frame is degenerate and the deflection is simply the new
    # direction, carrying the sign of the axis the photon was travelling along.
    sign = xp.where(uz >= 0, 1.0, -1.0)
    degenerate = xp.stack(
        [
            sin_theta * cos_phi,
            sign * sin_theta * sin_phi,
            sign * cos_theta,
        ],
        axis=-1,
    )
    at_pole = (1.0 - xp.abs(uz)) < _POLE_TOLERANCE
    return xp.where(at_pole[..., None], degenerate, general)


def run_transport_3d(
    backend: Backend,
    extinction_coefficient: float,
    single_scattering_albedo: float,
    asymmetry: float,
    config: TransportConfig | None = None,
    incidence: str | float = "collimated",
    surface_index: float = 1.31,
    ambient_index: float = 1.0,
    radial_edges_m: np.ndarray | None = None,
    phase: Any = None,
    obj: BuriedObject | None = None,
    keep_directions: bool = False,
) -> Transport3DResult:
    """Trace photons through a semi-infinite medium with a Fresnel surface.

    Args:
        backend: Array backend. All photon state lives on its device.
        extinction_coefficient: ``beta``, m^-1.
        single_scattering_albedo: ``omega``.
        asymmetry: ``g``, used when ``phase`` is not given.
        config: Run parameters, shared with the plane-parallel engine.
        incidence: ``"collimated"``, ``"diffuse"``, or a direction cosine.
        surface_index: Index of the medium. ``1.0`` matches the ambient and
            turns the interface off, which is what makes the plane-parallel
            engine a valid cross-check.
        ambient_index: Index above the surface.
        radial_edges_m: Bin edges for the radial profile.
        phase: A :class:`~snow_mcrt.domain.phase.TabulatedPhaseFunction`, or
            ``None`` to sample Henyey-Greenstein with ``asymmetry``. The
            tabulated Mie function is the physical one; HG is what the 1-D
            engine uses, so it is what a cross-check between them must use.
        keep_directions: Return the directions of surviving photons, for tests
            that assert on the geometry rather than on the physics.

    Returns:
        The energy ledger and the radially resolved reflectance.
    """
    config = config or TransportConfig()
    xp = backend.xp

    omega = float(single_scattering_albedo)
    g = float(asymmetry)
    beta = float(extinction_coefficient)
    if not 0 <= omega <= 1:
        raise ValueError("single-scattering albedo must lie in [0, 1]")
    if not -1 < g < 1:
        raise ValueError("asymmetry parameter must lie in (-1, 1)")
    if beta <= 0:
        raise ValueError("extinction coefficient must be positive")
    if surface_index < 1.0 or ambient_index < 1.0:
        raise ValueError("a refractive index below one is unphysical here")

    if config.delta_scaled and phase is None:
        f = g**2
        scaled_omega, scaled_g = delta_eddington_scaling(omega, g)
        omega, g = float(scaled_omega), float(scaled_g)
        beta = beta * (1.0 - f * single_scattering_albedo)

    if radial_edges_m is None:
        # A default spanning four decades around the transport mean free path,
        # which is the only length in the problem before omega is known.
        mfp = 1.0 / (beta * (1.0 - g))
        radial_edges_m = log_radial_edges(mfp * 1e-2, mfp * 1e2, 48)
    edges = np.asarray(radial_edges_m, dtype=float)
    n_bins = edges.size - 1
    edges_device = backend.asarray(edges)

    n = config.n_photons
    generator = backend.rng(config.seed)

    position = xp.zeros((n, 3))
    direction = _initial_directions(backend, generator, n, incidence)
    weight = xp.ones(n)
    alive = xp.ones(n, dtype=bool)
    # Which medium each photon is in. A photon starts at the surface, so it
    # starts in the snow whatever the object is.
    inside_object = xp.zeros(n, dtype=bool)

    if obj is None:
        beta_in, omega_in, g_in, index_in = beta, omega, g, surface_index
    else:
        beta_in = obj.extinction_coefficient
        omega_in = obj.single_scattering_albedo
        g_in = obj.asymmetry
        index_in = obj.refractive_index

    binned = xp.zeros(n_bins)
    reflected = 0.0
    outside = 0.0
    absorbed = 0.0
    scatters = 0

    matched = surface_index == ambient_index
    # The surface normal seen by a photon on its way out. z is depth, so a
    # photon leaving has u_z < 0 and the normal oriented against it is +z.
    up = backend.asarray(np.array([0.0, 0.0, 1.0]))

    for _ in range(config.max_scatters):
        if not bool(backend.to_numpy(alive.any())):
            break
        scatters += 1

        # Optical properties follow the photon, not the run. Where they are
        # uniform this collapses to the scalars it started as.
        beta_p = xp.where(inside_object, beta_in, beta)
        omega_p = xp.where(inside_object, omega_in, omega)
        g_p = xp.where(inside_object, g_in, g)

        u = backend.random_uniform(generator, (n,))
        # beta = 0 is a void: nothing to collide with, so the free path is
        # infinite and the photon always reaches a face first. Dividing by it
        # is the obvious way to turn a cavity into a nan.
        free_path = xp.where(
            beta_p > 0,
            -xp.log(xp.maximum(u, 1e-300)) / xp.maximum(beta_p, 1e-300),
            xp.inf,
        )

        # Distance to the surface along the current direction. Only photons
        # heading upward can reach it; for the rest it is unreachable and the
        # free path always wins.
        uz = direction[:, 2]
        heading_up = uz < 0.0
        to_surface = xp.where(
            heading_up, position[:, 2] / xp.maximum(-uz, 1e-300), xp.inf
        )

        if obj is None:
            to_object = xp.full(n, xp.inf)
            object_normal = None
        else:
            to_object, object_normal = obj.box.distance_to_surface(
                position, direction, xp=xp
            )

        # Whichever interface is nearer, and then only if the free path does
        # not get there first. Three competitors, one minimum.
        object_first = to_object < to_surface
        to_interface = xp.minimum(to_surface, to_object)
        reaches_interface = alive & (to_interface < free_path)
        hits_object = reaches_interface & object_first
        hits_surface = reaches_interface & ~object_first
        step = xp.where(reaches_interface, to_interface, free_path)
        position = xp.where(
            alive[:, None], position + step[:, None] * direction, position
        )

        if bool(backend.to_numpy(hits_surface.any())):
            cos_incident = -direction[:, 2]
            if matched:
                escapes = hits_surface
            else:
                r = fresnel_reflectance(
                    cos_incident, surface_index, ambient_index, xp=xp
                )
                draw = backend.random_uniform(generator, (n,))
                escapes = hits_surface & (draw >= r)
                bounced = hits_surface & ~escapes
                # Reflected photons stay in the medium, mirrored about the
                # surface. They sit exactly at z = 0 now heading down, so the
                # next iteration finds the surface unreachable, as it should.
                direction = xp.where(
                    bounced[:, None], specular_reflect(direction, up, xp=xp), direction
                )

            escaped_weight = xp.where(escapes, weight, 0.0)
            reflected += float(backend.to_numpy(escaped_weight.sum()))

            rho = xp.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
            index = xp.searchsorted(edges_device, rho, side="right") - 1
            inside = escapes & (index >= 0) & (index < n_bins)
            binned = binned + xp.bincount(
                xp.where(inside, index, 0),
                weights=xp.where(inside, escaped_weight, 0.0),
                minlength=n_bins,
            )[:n_bins]
            outside += float(
                backend.to_numpy(xp.where(escapes & ~inside, escaped_weight, 0.0).sum())
            )
            alive = alive & ~escapes

        if obj is not None and bool(backend.to_numpy(hits_object.any())):
            # Which way across the face, and therefore which pair of indices.
            # A photon inside is leaving; one outside is entering.
            n_from = xp.where(inside_object, index_in, surface_index)
            n_to = xp.where(inside_object, surface_index, index_in)
            cos_incident = -xp.sum(direction * object_normal, axis=-1)

            if index_in == surface_index:
                # Matched faces are not an interface at all. Skipping the
                # draw here keeps a matched object bit-identical to no object
                # in everything except where the steps were split.
                crosses = hits_object
            else:
                reflectance = _fresnel_pairwise(cos_incident, n_from, n_to, xp)
                draw = backend.random_uniform(generator, (n,))
                crosses = hits_object & (draw >= reflectance)
                turned = hits_object & ~crosses
                direction = xp.where(
                    turned[:, None],
                    specular_reflect(direction, object_normal, xp=xp),
                    direction,
                )
                direction = xp.where(
                    crosses[:, None],
                    _refract_pairwise(direction, object_normal, n_from, n_to, xp),
                    direction,
                )
            # The region changes only for photons that actually got through.
            inside_object = xp.where(crosses, ~inside_object, inside_object)

        # Only photons that reached a scattering event interact. A photon that
        # spent its step arriving at the surface has not yet met a grain.
        interacts = alive & ~reaches_interface
        deposited = xp.where(interacts, weight * (1.0 - omega_p), 0.0)
        absorbed += float(backend.to_numpy(deposited.sum()))
        weight = xp.where(interacts, weight * omega_p, weight)

        if config.roulette_threshold > 0:
            faint = alive & (weight < config.roulette_threshold)
            u_roulette = backend.random_uniform(generator, (n,))
            killed = faint & (u_roulette >= config.roulette_survival)
            survived = faint & ~killed
            weight = xp.where(survived, weight / config.roulette_survival, weight)
            alive = alive & ~killed

        if phase is None:
            cos_theta = _sample_hg_cosines_array(backend, generator, g_p, n)
        else:
            cos_theta = phase.sample(backend, generator, (n,))
        phi = 2.0 * np.pi * backend.random_uniform(generator, (n,))
        direction = xp.where(
            interacts[:, None], _rotate(backend, direction, cos_theta, phi), direction
        )

    truncated = float(backend.to_numpy(xp.where(alive, weight, 0.0).sum()))

    return Transport3DResult(
        reflected=reflected / n,
        absorbed=absorbed / n,
        truncated=truncated / n,
        binned_weight=backend.to_numpy(binned) / n,
        outside_bins=outside / n,
        edges_m=edges,
        scatters=scatters,
        config=config,
        surface_index=surface_index,
        final_directions=(
            backend.to_numpy(direction) if keep_directions else None
        ),
    )
