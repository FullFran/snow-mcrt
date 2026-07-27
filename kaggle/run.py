"""Kaggle batch job: validate the CuPy backend and run what a CPU cannot.

Two things need a GPU, and neither is Mie theory.

**The CuPy adapter has never run numerically.** On a machine without CUDA its
four numerical tests skip, leaving only the contract tests. An adapter that has
never executed is not an adapter, it is an intention.

**Clean snow in the visible is out of reach of one CPU core.** Transport cost
scales as ``1/(1 - omega)``, and clean snow at 500 nm sits at ``1 - omega ~
5e-6``. Measured locally: 200 000 scattering orders took 260 s for a case two
orders of magnitude *easier* than this one.

What does *not* belong here: the Mie and spectral albedo sweeps. Those are
CPU-bound -- ``miepython`` has no GPU backend -- and a Kaggle CPU is no faster
than a laptop. Sending them here would cost upload time and buy nothing.

Push it with::

    kaggle kernels push -p kaggle
    kaggle kernels status fullfran/snow-mcrt-gpu
    kaggle kernels output fullfran/snow-mcrt-gpu -p results/kaggle

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
    for omega in (0.5, 0.9, 0.99):
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
                    n_photons=200_000, seed=1, max_scatters=40_000
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
    for wavelength in (500.0, 700.0, 900.0):
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
                n_photons=1_000_000, seed=1, max_scatters=5_000_000
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


def _constants_path() -> str:
    """The ice constants fetched alongside the install."""
    if not CONSTANTS.exists():
        raise FileNotFoundError(f"ice optical constants not found at {CONSTANTS}")
    return str(CONSTANTS)


def main() -> int:
    install()
    manifest = {"environment": report_environment()}
    manifest.update(cross_check_backends())
    manifest.update(clean_snow_visible())
    (OUTPUT / "gpu-validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"written to {OUTPUT / 'gpu-validation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
