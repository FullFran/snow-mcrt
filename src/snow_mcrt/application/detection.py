"""How much an object changes the light coming back, and how fast that fades.

The detectability note answers this with diffusion theory and a two-way
attenuation argument. This answers it with transport: bury an object, trace
photons, and measure what changed. No closure assumption, no separability
assumption, and no assumption that the object is a small perturbation.

The quantity is **contrast**, the fractional change in returned light:

.. math::
    C = \\frac{R_\\text{object} - R_\\text{plain}}{R_\\text{plain}}

Negative when the object takes light away. It is a ratio on purpose: an
instrument measures a change against a background, not an absolute radiance,
and the ratio is what survives an unknown source brightness.

**Depth is reported in penetration depths, not centimetres.** A depth in
centimetres is a fact about one snowpack; the same object at the same
centimetre depth is trivially visible in clean polar snow and invisible in
dirty alpine snow, because the two differ by a factor of thirty in how far
light reaches. Quoted in ``delta`` the curve transfers, and it is also the
form the two-way argument in the note predicts.

Every depth costs a full transport run, and the plain reference is run once
and shared -- the same seed, so the two differ by the object and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snow_mcrt.domain.diffusion import DiffusionParameters
from snow_mcrt.domain.geometry import Box
from snow_mcrt.domain.transport import TransportConfig
from snow_mcrt.domain.transport3d import (
    BuriedObject,
    log_radial_edges,
    run_transport_3d,
)
from snow_mcrt.ports.backend import Backend


@dataclass(frozen=True)
class ContrastProfile:
    """Contrast against source-detector separation, at one burial depth.

    This is the measurement, and the total return is the shadow of it. Light
    that comes back close to the source never went deep enough to meet
    anything; light that comes back far away had to travel through the depth
    the object occupies. **The separation is a depth selector**, which is the
    whole physical basis of the method, and integrating over it throws that
    away.

    Args:
        rho_m: Separation at each bin centre, metres.
        plain: ``R(rho)`` with nothing buried, m^-2.
        with_object: ``R(rho)`` with the object, m^-2.
        mean_path_plain_m: Mean optical path returning at each separation
            with nothing buried.
        mean_path_object_m: The same with the object.
        n_photons: Photons per run, which sets the per-bin noise.
        penetration_depth_m: ``delta`` of the snowpack.
        transport_mfp_m: ``1 / (mu_a + mu_s')``.
        depth_m: Depth of the object's top face.
    """

    rho_m: np.ndarray
    plain: np.ndarray
    with_object: np.ndarray
    mean_path_plain_m: np.ndarray
    mean_path_object_m: np.ndarray
    n_photons: int
    penetration_depth_m: float
    transport_mfp_m: float
    depth_m: float

    @property
    def sampled(self) -> np.ndarray:
        """Bins both runs actually reached. An empty bin is not a zero."""
        return (self.plain > 0) & (self.with_object >= 0)

    @property
    def contrast(self) -> np.ndarray:
        """``(R_object - R_plain) / R_plain`` at each separation."""
        return np.divide(
            self.with_object - self.plain,
            self.plain,
            out=np.full_like(self.plain, np.nan),
            where=self.plain > 0,
        )

    @property
    def sigma(self) -> np.ndarray:
        """One-sigma noise on the contrast, bin by bin.

        Both runs are Poisson in the weight arriving, and the ratio carries
        both. This is what makes a far bin's spectacular contrast worthless:
        the signal grows towards 100% while the bin empties, and only the
        product of the two decides anything.
        """
        total = self.plain + self.with_object
        return np.divide(
            np.sqrt(np.maximum(total, 0.0) / self.n_photons),
            self.plain,
            out=np.full_like(self.plain, np.inf),
            where=self.plain > 0,
        )

    @property
    def snr(self) -> np.ndarray:
        """``|contrast| / sigma``. What actually decides a detection."""
        return np.abs(self.contrast) / self.sigma

    @property
    def rho_in_penetration_depths(self) -> np.ndarray:
        return self.rho_m / self.penetration_depth_m

    @property
    def best(self) -> int:
        """Index of the separation that detects best.

        Not the largest contrast -- that is always the farthest bin, where
        the object blocks everything and almost nothing arrives.
        """
        snr = np.where(np.isfinite(self.snr), self.snr, -np.inf)
        return int(np.argmax(snr))

    @property
    def path_contrast(self) -> np.ndarray:
        """Fractional change in mean optical path at each separation.

        The second, independent channel. An absorber removes the light that
        went deepest, so the light that survives came back *sooner* -- the
        mean path falls even where the intensity change is ambiguous. A
        time-resolved instrument measures this and the intensity separately.
        """
        return np.divide(
            self.mean_path_object_m - self.mean_path_plain_m,
            self.mean_path_plain_m,
            out=np.full_like(self.plain, np.nan),
            where=np.isfinite(self.mean_path_plain_m)
            & (self.mean_path_plain_m > 0),
        )

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "rho_m": self.rho_m,
            "rho_in_penetration_depths": self.rho_in_penetration_depths,
            "reflectance_plain_per_m2": self.plain,
            "reflectance_object_per_m2": self.with_object,
            "contrast": self.contrast,
            "sigma": self.sigma,
            "snr": self.snr,
            "mean_path_plain_m": self.mean_path_plain_m,
            "mean_path_object_m": self.mean_path_object_m,
            "path_contrast": self.path_contrast,
        }


@dataclass(frozen=True)
class DepthSweep:
    """Contrast against burial depth for one kind of object.

    Args:
        label: What was buried.
        depth_m: Depth of the object's top face, metres.
        contrast: Fractional change in returned light at each depth.
        reflected: Absolute return at each depth.
        plain_reflected: The reference with no object.
        penetration_depth_m: ``1 / mu_eff`` of the snowpack.
        transport_mfp_m: ``1 / (mu_a + mu_s')``.
        n_photons: Photons per point, which sets the noise floor.
    """

    label: str
    depth_m: np.ndarray
    contrast: np.ndarray
    reflected: np.ndarray
    plain_reflected: float
    penetration_depth_m: float
    transport_mfp_m: float
    n_photons: int

    @property
    def depth_in_penetration_depths(self) -> np.ndarray:
        """Burial depth in units of ``delta``. The transferable coordinate."""
        return self.depth_m / self.penetration_depth_m

    @property
    def noise_floor(self) -> float:
        """Smallest contrast this photon count can resolve.

        Two independent runs of a binomial proportion, so the difference
        carries ``sqrt(2)`` times the single-run error. A contrast below this
        is not a small detection, it is no detection -- and on a log axis it
        will still draw a perfectly convincing line.
        """
        p = self.plain_reflected
        return float(np.sqrt(2.0 * p * (1.0 - p) / self.n_photons) / p)

    @property
    def detectable(self) -> np.ndarray:
        """Where the contrast clears three times the noise floor."""
        return np.abs(self.contrast) > 3.0 * self.noise_floor

    def columns(self) -> dict[str, np.ndarray]:
        """Column name to values, in the order they are written to CSV."""
        return {
            "depth_m": self.depth_m,
            "depth_in_penetration_depths": self.depth_in_penetration_depths,
            "reflected": self.reflected,
            "contrast": self.contrast,
            "detectable": self.detectable.astype(int),
            # Constant down the column, and written anyway. A figure that has
            # to reconstruct the threshold from which points were flagged can
            # only bracket it, and would draw the shaded floor in slightly
            # the wrong place -- which is exactly the kind of quiet error
            # this column exists to prevent.
            "noise_floor": np.full_like(self.contrast, self.noise_floor),
        }


def sweep_contrast_profiles(
    backend: Backend,
    single_scattering_albedo: float,
    asymmetry: float,
    extinction_coefficient: float,
    depths_m: np.ndarray,
    half_width_m: float = 0.10,
    thickness_m: float = 0.20,
    object_extinction: float = 1e5,
    object_albedo: float = 0.0,
    object_index: float = 1.31,
    config: TransportConfig | None = None,
    surface_index: float = 1.31,
    inner_mfp: float = 1.0,
    outer_depths: float = 6.0,
    n_bins: int = 14,
) -> list[ContrastProfile]:
    """One profile per burial depth, sharing a single reference run.

    The snowpack with nothing in it does not depend on where the object would
    have been, so running it once per depth was pure waste. Worse, it put an
    independent realisation of the reference under every point, adding noise
    to a comparison whose entire job is to be quiet: with a shared reference
    the only thing that differs between two depths is the object.

    Halves the cost of a sweep, which matters because each run is a full
    transport solution.

    Returns:
        A profile per entry in ``depths_m``, in that order.
    """
    config = config or TransportConfig()
    parameters = DiffusionParameters.from_optical_properties(
        single_scattering_albedo,
        asymmetry,
        extinction_coefficient,
        refractive_index=surface_index,
    )
    edges = log_radial_edges(
        inner_mfp * parameters.transport_mean_free_path,
        outer_depths * parameters.penetration_depth,
        n_bins,
    )

    def trace(obj):
        return run_transport_3d(
            backend,
            extinction_coefficient,
            single_scattering_albedo,
            asymmetry,
            config=config,
            incidence="collimated",
            surface_index=surface_index,
            radial_edges_m=edges,
            obj=obj,
        )

    plain = trace(None)
    profiles = []
    for depth in np.atleast_1d(np.asarray(depths_m, dtype=float)):
        buried = trace(
            BuriedObject(
                Box(
                    lower=np.array([-half_width_m, -half_width_m, depth]),
                    upper=np.array(
                        [half_width_m, half_width_m, depth + thickness_m]
                    ),
                ),
                extinction_coefficient=object_extinction,
                single_scattering_albedo=object_albedo,
                refractive_index=object_index,
            )
        )
        profiles.append(
            ContrastProfile(
                rho_m=plain.bin_centres_m,
                plain=plain.reflectance,
                with_object=buried.reflectance,
                mean_path_plain_m=plain.mean_path_m,
                mean_path_object_m=buried.mean_path_m,
                n_photons=config.n_photons,
                penetration_depth_m=parameters.penetration_depth,
                transport_mfp_m=parameters.transport_mean_free_path,
                depth_m=float(depth),
            )
        )
    return profiles


def measure_contrast_profile(
    backend: Backend,
    single_scattering_albedo: float,
    asymmetry: float,
    extinction_coefficient: float,
    depth_m: float,
    half_width_m: float = 0.10,
    thickness_m: float = 0.20,
    object_extinction: float = 1e5,
    object_albedo: float = 0.0,
    object_index: float = 1.31,
    config: TransportConfig | None = None,
    surface_index: float = 1.31,
    inner_mfp: float = 1.0,
    outer_depths: float = 8.0,
    n_bins: int = 16,
) -> ContrastProfile:
    """Run the snowpack with and without the object, resolved by separation.

    Both runs share a seed, so a difference between them is the object rather
    than the realisation.

    Args:
        depth_m: Depth of the object's top face.
        inner_mfp: Innermost bin, in transport mean free paths.
        outer_depths: Outermost bin, in penetration depths.
        n_bins: Radial bins.

    Returns:
        The two profiles, their contrast, its noise, and the path lengths.
    """
    config = config or TransportConfig()
    parameters = DiffusionParameters.from_optical_properties(
        single_scattering_albedo,
        asymmetry,
        extinction_coefficient,
        refractive_index=surface_index,
    )
    edges = log_radial_edges(
        inner_mfp * parameters.transport_mean_free_path,
        outer_depths * parameters.penetration_depth,
        n_bins,
    )

    def trace(obj):
        return run_transport_3d(
            backend,
            extinction_coefficient,
            single_scattering_albedo,
            asymmetry,
            config=config,
            incidence="collimated",
            surface_index=surface_index,
            radial_edges_m=edges,
            obj=obj,
        )

    plain = trace(None)
    buried = trace(
        BuriedObject(
            Box(
                lower=np.array([-half_width_m, -half_width_m, depth_m]),
                upper=np.array(
                    [half_width_m, half_width_m, depth_m + thickness_m]
                ),
            ),
            extinction_coefficient=object_extinction,
            single_scattering_albedo=object_albedo,
            refractive_index=object_index,
        )
    )
    return ContrastProfile(
        rho_m=plain.bin_centres_m,
        plain=plain.reflectance,
        with_object=buried.reflectance,
        mean_path_plain_m=plain.mean_path_m,
        mean_path_object_m=buried.mean_path_m,
        n_photons=config.n_photons,
        penetration_depth_m=parameters.penetration_depth,
        transport_mfp_m=parameters.transport_mean_free_path,
        depth_m=depth_m,
    )


def sweep_burial_depth(
    backend: Backend,
    single_scattering_albedo: float,
    asymmetry: float,
    extinction_coefficient: float,
    depths_m: np.ndarray,
    label: str,
    half_width_m: float = 0.10,
    thickness_m: float = 0.20,
    object_extinction: float = 1e5,
    object_albedo: float = 0.0,
    object_index: float = 1.31,
    config: TransportConfig | None = None,
    surface_index: float = 1.31,
) -> DepthSweep:
    """Trace the same object at a series of burial depths.

    Args:
        backend: Array backend.
        single_scattering_albedo: ``omega`` of the snow.
        asymmetry: ``g`` of the snow.
        extinction_coefficient: ``beta`` of the snow, m^-1.
        depths_m: Depths of the object's top face.
        label: What is being buried, carried into the result.
        half_width_m: Half the object's horizontal extent.
        thickness_m: Its vertical extent.
        object_extinction: ``beta`` inside it. Zero makes it a void.
        object_albedo: ``omega`` inside it. Zero makes it black.
        object_index: Its refractive index.
        config: Transport parameters, shared by every point including the
            reference, so a difference is the object and not the seed.
        surface_index: Index of the snow.

    Returns:
        Contrast against depth.
    """
    config = config or TransportConfig()
    parameters = DiffusionParameters.from_optical_properties(
        single_scattering_albedo,
        asymmetry,
        extinction_coefficient,
        refractive_index=surface_index,
    )

    def trace(obj: BuriedObject | None) -> float:
        return run_transport_3d(
            backend,
            extinction_coefficient,
            single_scattering_albedo,
            asymmetry,
            config=config,
            incidence="collimated",
            surface_index=surface_index,
            obj=obj,
        ).reflected

    plain = trace(None)
    if plain <= 0:
        raise ValueError(
            "the snowpack returns no light at all, so contrast is undefined"
        )

    depths = np.atleast_1d(np.asarray(depths_m, dtype=float))
    returned = np.array(
        [
            trace(
                BuriedObject(
                    Box(
                        lower=np.array([-half_width_m, -half_width_m, depth]),
                        upper=np.array(
                            [half_width_m, half_width_m, depth + thickness_m]
                        ),
                    ),
                    extinction_coefficient=object_extinction,
                    single_scattering_albedo=object_albedo,
                    refractive_index=object_index,
                )
            )
            for depth in depths
        ]
    )
    return DepthSweep(
        label=label,
        depth_m=depths,
        contrast=(returned - plain) / plain,
        reflected=returned,
        plain_reflected=plain,
        penetration_depth_m=parameters.penetration_depth,
        transport_mfp_m=parameters.transport_mean_free_path,
        n_photons=config.n_photons,
    )
