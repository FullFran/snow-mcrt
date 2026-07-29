"""Kaggle batch job: validate the CuPy backend and run what a CPU cannot.

Two things need a GPU, and neither is Mie theory.

**The CuPy adapter has never run numerically.** On a machine without CUDA its
four numerical tests skip, leaving only the contract tests. An adapter that has
never executed is not an adapter, it is an intention.

**High-albedo transport is out of reach of one CPU core.** Cost scales as
``1/(1 - omega)``. Measured locally: 200 000 scattering orders took 260 s at
``1 - omega ~ 5e-5``, with 20 000 photons.

The budgets below are deliberately finite. Clean snow at 500 nm sits at
``1 - omega ~ 5e-6`` and would need of order a million scattering orders; at
500 000 photons that is over an hour of wall clock even on a GPU, which is a
benchmark of patience rather than of physics. So this runs 700-1100 nm, where
the answer is reachable, and reports ``truncated`` so any shortfall is visible
rather than absorbed into a plausible-looking albedo.

What does *not* belong here: the Mie and spectral albedo sweeps. Those are
CPU-bound -- ``miepython`` has no GPU backend -- and a Kaggle CPU is no faster
than a laptop. Sending them here would cost upload time and buy nothing.

Push it with::

    kaggle kernels push -p kaggle
    kaggle kernels status fran17/snow-mcrt-gpu-validation
    kaggle kernels output fran17/snow-mcrt-gpu-validation -p results/kaggle

The repository is public, so the install is a plain pip from GitHub. No
vendored copy, which would drift out of sync with the branch it claims to be.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "github.com/FullFran/snow-mcrt.git"
BRANCH = "main"
OUTPUT = Path("/kaggle/working")
CONSTANTS = OUTPUT / "warren_brandt_2008.dat"


def install() -> None:
    """Install the package and fetch the ice constants it needs."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"git+https://{REPO}@{BRANCH}"],
        check=True,
    )
    # pip installs the package, not the repository data directory, so the
    # optical constants are fetched separately rather than vendored into the
    # wheel where they would be invisible to anyone auditing provenance.
    subprocess.run(
        [
            "curl", "-sSL", "-o", str(CONSTANTS),
            f"https://raw.githubusercontent.com/FullFran/snow-mcrt/{BRANCH}"
            "/data/ice/warren_brandt_2008.dat",
        ],
        check=True,
    )


def report_environment() -> dict:
    """Record what we actually ran on. A benchmark without this is a rumour."""
    import cupy as cp

    device = cp.cuda.runtime.getDeviceProperties(0)
    info = {
        "gpu": device["name"].decode(),
        "cupy": cp.__version__,
        "total_memory_gb": round(device["totalGlobalMem"] / 1024**3, 2),
    }
    print(json.dumps(info, indent=2), flush=True)
    return info


def cross_check_backends() -> dict:
    """NumPy is the oracle. If CuPy disagrees with it, CuPy is wrong.

    Both backends run the same transport at the same seed. They will not agree
    bit for bit -- the generators differ -- but they must agree statistically,
    and a real disagreement means the adapter is broken.
    """
    from snow_mcrt.adapters.cupy_backend import CupyBackend
    from snow_mcrt.adapters.numpy_backend import NumpyBackend
    from snow_mcrt.domain.analytic import van_de_hulst_semi_infinite_albedo
    from snow_mcrt.domain.transport import TransportConfig, run_transport

    results = []
    for omega in (0.5, 0.9, 0.95):
        row = {"omega": omega, "reference": float(
            van_de_hulst_semi_infinite_albedo(omega)
        )}
        for name, backend in (("numpy", NumpyBackend()), ("cupy", CupyBackend())):
            started = time.time()
            result = run_transport(
                backend,
                5000.0,
                omega,
                0.0,
                config=TransportConfig(
                    n_photons=100_000, seed=1, max_scatters=20_000
                ),
            )
            row[name] = result.albedo
            row[f"{name}_seconds"] = round(time.time() - started, 2)
        row["relative_difference"] = abs(row["cupy"] - row["numpy"]) / row["numpy"]
        results.append(row)
        print(json.dumps(row), flush=True)
    return {"backend_cross_check": results}


def clean_snow_visible() -> dict:
    """The case a CPU cannot reach: research question 1 at its hardest point.

    Clean snow at 500 nm. The analytic answer is known, so this is a test of
    whether the transport can actually get there, not of what the answer is.
    """
    import numpy as np

    from snow_mcrt.adapters.cupy_backend import CupyBackend
    from snow_mcrt.adapters.miepython_solver import MiepythonSolver
    from snow_mcrt.adapters.tabulated_constants import TabulatedConstants
    from snow_mcrt.domain.analytic import similarity_scaled_albedo
    from snow_mcrt.domain.medium import SnowLayer, compute_layer_properties
    from snow_mcrt.domain.transport import TransportConfig, run_transport

    constants = TabulatedConstants(
        _constants_path(), name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()
    solver = MiepythonSolver()
    backend = CupyBackend()

    rows = []
    for wavelength in (700.0, 900.0, 1100.0):
        layer = SnowLayer(100e-6, 300.0)
        props = compute_layer_properties(
            solver, layer, constants.m_at(wavelength), wavelength
        )
        omega = float(props.single_scattering_albedo[0])
        g = float(props.asymmetry[0])
        beta = float(props.extinction_coefficient[0])

        started = time.time()
        result = run_transport(
            backend,
            beta,
            omega,
            g,
            config=TransportConfig(
                n_photons=500_000, seed=1, max_scatters=300_000
            ),
        )
        expected = float(similarity_scaled_albedo(omega, g))
        row = {
            "wavelength_nm": wavelength,
            "co_albedo": 1.0 - omega,
            "monte_carlo": result.albedo,
            "analytic": expected,
            "relative_difference": (result.albedo - expected) / expected,
            "truncated": result.truncated,
            "scatters": result.scatters,
            "seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"clean_snow_visible": rows}


def diffusion_validity() -> dict:
    """How far diffusion theory departs from transport, at a reachable count.

    Every figure in ``docs/detectability.md`` is a diffusion calculation. This
    measures the approximation against the 3-D engine, which makes no closure
    assumption at all.

    The reason it belongs on a GPU rather than a laptop is the *tail*. The
    profile falls seven orders of magnitude over the range that matters, so
    the far bins are starved of photons long before the near ones are, and a
    starved bin is not a small measurement -- it is no measurement. Clean snow
    at 450 nm makes this acute: ``1 - omega`` is about 3e-7, so a photon takes
    of order a million scattering orders to be absorbed, and locally the run
    ends with several percent of the weight still in flight. ``truncated`` is
    reported for exactly that reason.

    Writes the profiles as CSV as well as the summary, because the ratio bin
    by bin is the result; the worst-case number is only its headline.
    """
    import csv

    import numpy as np

    from snow_mcrt.adapters.cupy_backend import CupyBackend
    from snow_mcrt.adapters.miepython_solver import MiepythonSolver
    from snow_mcrt.adapters.tabulated_constants import TabulatedConstants
    from snow_mcrt.application.validate_diffusion import compare_with_diffusion
    from snow_mcrt.domain.medium import (
        BLACK_CARBON,
        ImpurityLoading,
        SnowLayer,
        compute_layer_properties,
    )
    from snow_mcrt.domain.transport import TransportConfig

    constants = TabulatedConstants(
        _constants_path(), name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()
    solver = MiepythonSolver()
    backend = CupyBackend()

    cases = (
        ("clean-450nm", 450.0, 0.0),
        ("arctic-450nm", 450.0, 10.0),
        ("alpine-450nm", 450.0, 100.0),
        ("alpine-800nm", 800.0, 100.0),
    )

    rows = []
    for label, wavelength, ng_per_g in cases:
        grid = np.array([wavelength])
        impurities = (
            (ImpurityLoading.from_ng_per_g(BLACK_CARBON, ng_per_g),)
            if ng_per_g
            else ()
        )
        props = compute_layer_properties(
            solver,
            SnowLayer(100e-6, 300.0, impurities=impurities),
            constants.m_at(grid),
            grid,
        )
        started = time.time()
        comparison = compare_with_diffusion(
            backend,
            float(props.single_scattering_albedo[0]),
            float(props.asymmetry[0]),
            float(props.extinction_coefficient[0]),
            config=TransportConfig(
                n_photons=2_000_000, seed=7, max_scatters=400_000
            ),
        )
        sampled = comparison.sampled()
        path = OUTPUT / f"mc-diffusion-{label}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            columns = comparison.columns()
            writer.writerow(columns.keys())
            for values in zip(*(np.asarray(v)[sampled] for v in columns.values())):
                writer.writerow(f"{value:.10g}" for value in values)

        row = {
            "case": label,
            "wavelength_nm": wavelength,
            "black_carbon_ng_per_g": ng_per_g,
            "co_albedo": 1.0 - comparison.single_scattering_albedo,
            "asymmetry": comparison.asymmetry,
            "transport_mfp_cm": comparison.transport_mfp_m * 100,
            "penetration_depth_cm": comparison.penetration_depth_m * 100,
            "reflected": comparison.reflected,
            "truncated": comparison.truncated,
            # The band a source-detector pair actually uses. Reported first
            # because the worst case inside 12 mfp' is dominated by the near
            # field, where diffusion is inapplicable rather than inaccurate.
            "departure_3_to_12_mfp": comparison.departure_between(3.0, 12.0),
            "departure_12_to_50_mfp": comparison.departure_between(12.0, 50.0),
            "worst_ratio_within_12_mfp": comparison.worst_ratio_within(12.0),
            "ratio_far_tail": float(
                np.nanmax(comparison.ratio[sampled])
            ),
            "bins_sampled": int(sampled.sum()),
            "seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"diffusion_validity": rows}


def detection_profiles() -> dict:
    """Contrast against source-detector separation, at several burial depths.

    The measurement, rather than the shadow of it. Total returned light
    dilutes the signal across a field most of which never met the object:
    photons coming back close to the source went shallowest and carry the
    least about anything buried. Separation is a depth selector.

    It belongs on a GPU for the same reason the diffusion comparison does,
    only more so. The profile has to be resolved *per depth*, so every bin
    needs enough photons on its own, and the far bins -- the ones that carry
    the signal -- are the starved ones. Clean visible snow needs of order a
    million scattering orders per photon on top of that.

    The reference snowpack is traced once and shared across depths: it does
    not depend on where the object would have been, and a shared reference
    means the only thing differing between two depths is the object.
    """
    import csv

    import numpy as np

    from snow_mcrt.adapters.cupy_backend import CupyBackend
    from snow_mcrt.adapters.miepython_solver import MiepythonSolver
    from snow_mcrt.adapters.tabulated_constants import TabulatedConstants
    from snow_mcrt.application.detection import iter_contrast_profiles
    from snow_mcrt.domain.diffusion import DiffusionParameters
    from snow_mcrt.domain.medium import (
        BLACK_CARBON,
        ImpurityLoading,
        SnowLayer,
        compute_layer_properties,
    )
    from snow_mcrt.domain.transport import TransportConfig

    constants = TabulatedConstants(
        _constants_path(), name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()
    solver = MiepythonSolver()
    backend = CupyBackend()

    grid = np.array([450.0])
    props = compute_layer_properties(
        solver,
        SnowLayer(
            100e-6,
            300.0,
            impurities=(ImpurityLoading.from_ng_per_g(BLACK_CARBON, 100.0),),
        ),
        constants.m_at(grid),
        grid,
    )
    omega = float(props.single_scattering_albedo[0])
    g = float(props.asymmetry[0])
    beta = float(props.extinction_coefficient[0])
    delta = DiffusionParameters.from_optical_properties(
        omega, g, beta, refractive_index=1.31
    ).penetration_depth

    def write(profile, tag: str) -> dict:
        """Write one profile and summarise it. Called the moment it exists."""
        path = OUTPUT / f"detection-{tag}-{profile.depth_m / delta:.2f}delta.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            columns = profile.columns()
            sampled = profile.sampled
            writer.writerow(columns.keys())
            for values in zip(*(np.asarray(v)[sampled] for v in columns.values())):
                writer.writerow(f"{value:.10g}" for value in values)

        best = profile.best
        integrated = float(
            (profile.with_object.sum() - profile.plain.sum()) / profile.plain.sum()
        )
        row = {
            "channel": tag,
            "depth_m": profile.depth_m,
            "depth_in_penetration_depths": profile.depth_m / delta,
            "contrast_integrated": integrated,
            "contrast_best": float(profile.contrast[best]),
            "best_rho_m": float(profile.rho_m[best]),
            "best_rho_in_penetration_depths": float(
                profile.rho_in_penetration_depths[best]
            ),
            "best_snr": float(profile.snr[best]),
            "path_contrast_at_best": float(profile.path_contrast[best]),
            "gain_over_integrating": abs(float(profile.contrast[best]) / integrated)
            if integrated
            else float("nan"),
            "seconds": round(time.time() - started, 1),
        }
        print(json.dumps(row), flush=True)
        return row

    started = time.time()
    rows = []

    def snapshot() -> dict:
        """What the manifest should say if the run stops right now.

        Carries the snowpack with the points, not just the points. A partial
        file listing contrasts against depths in metres, with no penetration
        depth to divide by, is a file nobody can read six months later.
        """
        return {
            "detection_profiles": {
                "penetration_depth_cm": delta * 100,
                "co_albedo": 1.0 - omega,
                "asymmetry": g,
                "complete": False,
                "seconds": round(time.time() - started, 1),
                "points": rows,
            }
        }

    # The intensity channel. Delta-scaled, because the contrast is a ratio of
    # intensities and does not care about path length -- and because at
    # g = 0.889 turning the scaling off multiplies the step count several
    # times over. Calibrating the cost of an unscaled run against scaled ones
    # is what overran the previous session by a factor nobody budgeted for.
    for profile in iter_contrast_profiles(
        backend,
        omega,
        g,
        beta,
        depths_m=np.array([0.15, 0.35, 0.6, 1.0, 1.5, 2.2]) * delta,
        config=TransportConfig(n_photons=500_000, seed=11, max_scatters=200_000),
    ):
        rows.append(write(profile, "intensity"))
        _checkpoint(snapshot())

    # The path channel, and only two depths of it. Path length is meaningless
    # under delta scaling -- the scaled medium reaches the same place in
    # fewer, longer steps -- so this one has to run unscaled, and unscaled is
    # expensive. Two depths bracket the useful range; six would not finish.
    for profile in iter_contrast_profiles(
        backend,
        omega,
        g,
        beta,
        depths_m=np.array([0.35, 1.0]) * delta,
        config=TransportConfig(
            n_photons=200_000, seed=11, max_scatters=400_000, delta_scaled=False
        ),
    ):
        rows.append(write(profile, "path"))
        _checkpoint(snapshot())

    final = snapshot()
    final["detection_profiles"]["complete"] = True
    return final


def _constants_path() -> str:
    """The ice constants fetched alongside the install."""
    if not CONSTANTS.exists():
        raise FileNotFoundError(f"ice optical constants not found at {CONSTANTS}")
    return str(CONSTANTS)


_MANIFEST = OUTPUT / "gpu-validation.json"
_STATE: dict = {}


def _checkpoint(update: dict) -> None:
    """Merge into the manifest and write it out, now.

    A kernel here is hours of GPU under a hard wall-clock limit, and Kaggle
    keeps the output of a session it cancelled. Writing the manifest once at
    the end therefore turns "ran out of time" into "produced nothing" -- which
    is exactly what happened on the previous attempt: the run completed five
    hours of work and delivered none of it.

    Cheap enough to call after every profile. The file is small and the run is
    not.
    """
    _STATE.update(update)
    _MANIFEST.write_text(json.dumps(_STATE, indent=2) + "\n")


# Which sections to run. The diffusion comparison is five hours of GPU whose
# results are already committed under results/kaggle/ and do not change, so
# re-running it only eats the budget the new work needs. Restore the full
# tuple when something upstream of it changes.
SECTIONS = ("detection",)

_ALL_SECTIONS = {
    "backends": cross_check_backends,
    "deep": clean_snow_visible,
    "diffusion": diffusion_validity,
    "detection": detection_profiles,
}


def main() -> int:
    install()
    _checkpoint({"environment": report_environment(), "sections": list(SECTIONS)})
    for name in SECTIONS:
        section = _ALL_SECTIONS[name]
        print(f"--- {name} ---", flush=True)
        _checkpoint(section())
    print(f"written to {OUTPUT / 'gpu-validation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
