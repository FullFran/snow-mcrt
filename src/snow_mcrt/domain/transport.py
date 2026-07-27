"""Vectorised Monte Carlo photon transport through a plane-parallel snowpack.

Photons are an **axis of the array**, not an iteration. Every live photon takes
its step, samples its deflection and updates its weight in the same handful of
array operations, and the loop runs once per scattering order rather than once
per photon. That is what makes a GPU backend worth anything here: a single
photon traced to completion is a different program, not a slower one.

Geometry is one-dimensional and azimuthally symmetric. A photon is fully
described by its depth ``z``, the cosine ``mu`` of its direction against the
downward normal, and its weight. There is no need to carry ``x``, ``y`` or an
azimuth: a plane-parallel medium is invariant under both, so tracking them
would be arithmetic that no observable ever reads.

**Implicit capture.** Photons are not killed by absorption; their weight is
multiplied by ``omega`` at each collision. Explicit absorption would be
hopeless here. Clean snow in the visible has ``1 - omega ~ 4e-6``, so a photon
survives some 200 000 collisions on average and the variance of any
absorption-terminated estimator is enormous. Weighting removes that noise
entirely and replaces it with a bias-free bookkeeping problem.

Even so this is genuinely expensive, and honestly so: light really does
scatter tens of thousands of times inside snow before it leaves. The cost is
physics, not implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from snow_mcrt.domain.analytic import delta_eddington_scaling
from snow_mcrt.ports.backend import Backend

_ISOTROPIC_G_THRESHOLD = 1e-6


@dataclass(frozen=True)
class TransportConfig:
    """Run parameters for one Monte Carlo simulation.

    Args:
        n_photons: Photons launched. They are traced simultaneously.
        seed: Seed for the backend's generator. Recorded in the manifest;
            identical configs must reproduce identical results.
        max_scatters: Hard cap on scattering orders. Weight still in flight
            when the cap is reached is reported separately rather than being
            quietly folded into absorption -- see
            :attr:`TransportResult.truncated`.
        roulette_threshold: Weight below which Russian roulette begins.
        roulette_survival: Survival probability in roulette. Survivors have
            their weight divided by it, which keeps the estimator unbiased.
        delta_scaled: Apply delta-Eddington scaling to the optical properties
            before transport. Strongly recommended: an ice grain has ``g~0.89``
            and truncating the forward peak roughly halves the number of
            scattering orders needed to reach the same answer.
    """

    n_photons: int = 100_000
    seed: int = 0
    max_scatters: int = 100_000
    roulette_threshold: float = 1e-4
    roulette_survival: float = 0.1
    delta_scaled: bool = True

    def __post_init__(self) -> None:
        if self.n_photons <= 0:
            raise ValueError("need at least one photon")
        if self.max_scatters <= 0:
            raise ValueError("max_scatters must be positive")
        if not 0 < self.roulette_survival < 1:
            raise ValueError("roulette survival probability must lie in (0, 1)")
        if self.roulette_threshold < 0:
            raise ValueError("roulette threshold must be non-negative")


@dataclass(frozen=True)
class TransportResult:
    """Outcome of one run, as fractions of the launched energy.

    Args:
        reflected: Weight that escaped upward through the surface. This is the
            albedo.
        transmitted: Weight that escaped through the bottom.
        absorbed: Weight deposited in the pack.
        truncated: Weight still in flight when ``max_scatters`` was reached.
            Reported rather than absorbed, because silently attributing it
            would turn a convergence failure into a plausible-looking result.
        scatters: Scattering orders actually executed.
        config: The configuration that produced this, for the manifest.
    """

    reflected: float
    transmitted: float
    absorbed: float
    truncated: float
    scatters: int
    config: TransportConfig

    @property
    def albedo(self) -> float:
        """Fraction of incident energy reflected back out of the surface."""
        return self.reflected

    @property
    def energy_balance(self) -> float:
        """Total accounted energy, which must come to 1.

        Exactly 1, to float64 epsilon, when Russian roulette is disabled --
        nothing is ever discarded, so the ledger is an identity.

        With roulette on it holds only *in expectation*. Killed photons take
        their weight with them and survivors are boosted by ``1/p`` to
        compensate; the two cancel on average, leaving a residual of order the
        roulette threshold whose sign changes between runs. That is the
        correct behaviour of an unbiased estimator, not a leak.
        """
        return self.reflected + self.transmitted + self.absorbed + self.truncated


def _sample_initial_mu(
    backend: Backend, generator: Any, n: int, incidence: str | float
) -> Any:
    """Downward direction cosines for the incident photons."""
    if incidence == "diffuse":
        # Lambertian incidence: the flux-weighted cosine distribution, whose
        # inverse CDF is sqrt(u). This is the illumination the two-stream
        # oracle assumes, so it is the default.
        return backend.xp.sqrt(backend.random_uniform(generator, (n,)))
    mu0 = float(incidence)
    if not 0 < mu0 <= 1:
        raise ValueError(
            "collimated incidence needs a direction cosine in (0, 1]; "
            "mu0 = 0 is grazing and never enters the medium"
        )
    return backend.xp.full((n,), mu0)


def _deflect(backend: Backend, mu: Any, cos_theta: Any, phi: Any) -> Any:
    """Rotate a direction cosine by a scattering angle.

    In a plane-parallel medium only the polar cosine survives the azimuthal
    average, so the full rotation collapses to

    .. math::
        \\mu' = \\mu \\cos\\theta +
                \\sqrt{1 - \\mu^2}\\sqrt{1 - \\cos^2\\theta}\\cos\\phi
    """
    xp = backend.xp
    sin_mu = xp.sqrt(xp.maximum(0.0, 1.0 - mu * mu))
    sin_theta = xp.sqrt(xp.maximum(0.0, 1.0 - cos_theta * cos_theta))
    return xp.clip(mu * cos_theta + sin_mu * sin_theta * xp.cos(phi), -1.0, 1.0)


def _sample_hg_cosines(backend: Backend, generator: Any, g: float, n: int) -> Any:
    """Henyey-Greenstein deflection cosines, drawn on the backend's device."""
    u = backend.random_uniform(generator, (n,))
    if abs(g) < _ISOTROPIC_G_THRESHOLD:
        return 2.0 * u - 1.0
    term = (1.0 - g**2) / (1.0 - g + 2.0 * g * u)
    return backend.xp.clip((1.0 + g**2 - term**2) / (2.0 * g), -1.0, 1.0)


def run_transport(
    backend: Backend,
    extinction_coefficient: float,
    single_scattering_albedo: float,
    asymmetry: float,
    thickness_m: float = np.inf,
    config: TransportConfig | None = None,
    incidence: str | float = "diffuse",
) -> TransportResult:
    """Trace photons through a homogeneous plane-parallel layer.

    Args:
        backend: Array backend. All photon state lives on its device.
        extinction_coefficient: ``beta``, m^-1.
        single_scattering_albedo: ``omega``.
        asymmetry: ``g``.
        thickness_m: Layer thickness. ``inf`` for a semi-infinite pack.
        config: Run parameters.
        incidence: ``"diffuse"`` for Lambertian illumination, matching the
            analytic oracle, or a direction cosine for a collimated beam.

    Returns:
        Reflected, transmitted, absorbed and truncated energy fractions.
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

    if config.delta_scaled:
        f = g**2
        scaled_omega, scaled_g = delta_eddington_scaling(omega, g)
        omega, g = float(scaled_omega), float(scaled_g)
        beta = beta * (1.0 - f * single_scattering_albedo)

    n = config.n_photons
    generator = backend.rng(config.seed)

    z = xp.zeros(n)
    mu = _sample_initial_mu(backend, generator, n, incidence)
    weight = xp.ones(n)
    alive = xp.ones(n, dtype=bool)

    reflected = 0.0
    transmitted = 0.0
    absorbed = 0.0
    scatters = 0

    for _ in range(config.max_scatters):
        n_alive = int(backend.to_numpy(alive.sum()))
        if n_alive == 0:
            break
        scatters += 1

        # Free path. Sampling -log(u)/beta is exact for an exponential, and
        # the whole array advances at once.
        u = backend.random_uniform(generator, (n,))
        step = -xp.log(xp.maximum(u, 1e-300)) / beta
        z = xp.where(alive, z + step * mu, z)

        escaped_up = alive & (z < 0.0)
        reflected += float(backend.to_numpy(xp.sum(xp.where(escaped_up, weight, 0.0))))
        alive = alive & ~escaped_up

        if np.isfinite(thickness_m):
            escaped_down = alive & (z > thickness_m)
            transmitted += float(
                backend.to_numpy(xp.sum(xp.where(escaped_down, weight, 0.0)))
            )
            alive = alive & ~escaped_down

        # Implicit capture: deposit the absorbed fraction, keep the photon.
        deposited = xp.where(alive, weight * (1.0 - omega), 0.0)
        absorbed += float(backend.to_numpy(xp.sum(deposited)))
        weight = xp.where(alive, weight * omega, weight)

        # Russian roulette. Survivors are boosted by 1/p so the estimator
        # stays unbiased; without the boost this would quietly lose energy.
        if config.roulette_threshold > 0:
            faint = alive & (weight < config.roulette_threshold)
            u_roulette = backend.random_uniform(generator, (n,))
            killed = faint & (u_roulette >= config.roulette_survival)
            survived = faint & ~killed
            weight = xp.where(survived, weight / config.roulette_survival, weight)
            alive = alive & ~killed

        cos_theta = _sample_hg_cosines(backend, generator, g, n)
        phi = 2.0 * np.pi * backend.random_uniform(generator, (n,))
        mu = xp.where(alive, _deflect(backend, mu, cos_theta, phi), mu)

    truncated = float(backend.to_numpy(xp.sum(xp.where(alive, weight, 0.0))))

    return TransportResult(
        reflected=reflected / n,
        transmitted=transmitted / n,
        absorbed=absorbed / n,
        truncated=truncated / n,
        scatters=scatters,
        config=config,
    )
