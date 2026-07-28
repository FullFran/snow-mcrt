#!/usr/bin/env python3
"""Generate the reference spectral albedo curves.

Headless by construction: this script computes and writes, and it draws
nothing. That is what lets it run unchanged inside a batch job with no display
attached, and it is why the figures can be regenerated from the committed CSVs
without re-running any physics.

Usage::

    python scripts/run_albedo.py --output data/reference
    python scripts/run_albedo.py --output /tmp/scratch --points 60
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snow_mcrt.adapters.cached_mie_solver import (  # noqa: E402
    CachedMieSolver,
    frozen_table_paths,
)
from snow_mcrt.adapters.miepython_solver import MiepythonSolver  # noqa: E402
from snow_mcrt.adapters.numpy_backend import NumpyBackend  # noqa: E402
from snow_mcrt.adapters.tabulated_constants import TabulatedConstants  # noqa: E402
from snow_mcrt.application.spectral_albedo import (  # noqa: E402
    SpectralAlbedoConfig,
    run_spectral_albedo,
)
from snow_mcrt.domain.transport import TransportConfig  # noqa: E402
from snow_mcrt.infra.writers import write_result  # noqa: E402

DEFAULT_CONSTANTS = (
    Path(__file__).resolve().parent.parent / "data" / "ice" / "warren_brandt_2008.dat"
)
# The committed Mie table. Read-only here: it is regenerated deliberately by
# scripts/build_mie_cache.py, never as a side effect of a run.
MIE_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "mie"

# Wiscombe & Warren (1980) span the solar spectrum. 300 nm is the short-wave
# end of the usable ice data; below 390 nm the tabulated k is a reported upper
# limit rather than a measurement -- see data/ice/SOURCE.md.
WAVELENGTH_RANGE_NM = (300.0, 2500.0)

GRAIN_RADII_UM = (50.0, 100.0, 250.0, 500.0, 1000.0)
BLACK_CARBON_NG_PER_G = (0.0, 1.0, 10.0, 100.0, 1000.0)


def wavelength_grid(points: int) -> np.ndarray:
    """Logarithmic grid, which spreads points where the physics moves."""
    return np.logspace(
        np.log10(WAVELENGTH_RANGE_NM[0]), np.log10(WAVELENGTH_RANGE_NM[1]), points
    )


def generate(args: argparse.Namespace, solver) -> int:
    """Write every reference curve, using the solver it is handed.

    Separated from argument parsing so the solver is chosen in exactly one
    place and this function never has to know whether it is cached.
    """
    constants = TabulatedConstants(
        args.constants, name="Warren & Brandt 2008", wavelength_scale_to_nm=1000.0
    ).load()
    wavelengths = wavelength_grid(args.points)

    print(f"optical constants : {constants.name}")
    print(f"tabulated range   : {constants.wavelength_range_nm[0]:.1f}"
          f" to {constants.wavelength_range_nm[1]:.1f} nm")
    print(f"grid              : {args.points} points, "
          f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")
    print()

    # Research question 1 -- pure snow across grain sizes.
    for radius_um in GRAIN_RADII_UM:
        config = SpectralAlbedoConfig(
            wavelengths_nm=wavelengths,
            grain_radius_m=radius_um * 1e-6,
            label=f"pure-r{radius_um:g}um",
        )
        result = run_spectral_albedo(solver, constants, config)
        csv_path, _ = write_result(result, args.output)
        print(f"  {csv_path.name:28} albedo {result.albedo.min():.4f}"
              f" to {result.albedo.max():.4f}")

    # Research question 2 -- black carbon at fixed grain size.
    for loading in BLACK_CARBON_NG_PER_G:
        config = SpectralAlbedoConfig(
            wavelengths_nm=wavelengths,
            grain_radius_m=100e-6,
            black_carbon_ng_per_g=loading,
            label=f"bc{loading:g}ngg-r100um",
        )
        result = run_spectral_albedo(solver, constants, config)
        csv_path, _ = write_result(result, args.output)
        print(f"  {csv_path.name:28} albedo {result.albedo.min():.4f}"
              f" to {result.albedo.max():.4f}")

    if args.monte_carlo:
        # A handful of wavelengths only. Cost scales as 1/(1-omega), so a full
        # visible sweep is not a slow run, it is an infeasible one.
        print()
        print("Monte Carlo spot check (transport engine vs analytic):")
        spot = np.array([1000.0, 1300.0, 1600.0, 2000.0])
        backend = NumpyBackend()
        mc_config = SpectralAlbedoConfig(
            wavelengths_nm=spot,
            grain_radius_m=100e-6,
            method="monte-carlo",
            transport=TransportConfig(n_photons=20_000, seed=1, max_scatters=100_000),
            label="mc-spotcheck-r100um",
        )
        mc = run_spectral_albedo(solver, constants, mc_config, backend=backend)
        analytic = run_spectral_albedo(
            solver,
            constants,
            SpectralAlbedoConfig(
                wavelengths_nm=spot, grain_radius_m=100e-6, label="analytic-at-spot"
            ),
        )
        write_result(mc, args.output)
        print(f"    {'lambda':>8} {'MC':>8} {'analytic':>9} {'rel':>8}")
        for i, lam in enumerate(spot):
            rel = (mc.albedo[i] - analytic.albedo[i]) / analytic.albedo[i]
            print(f"    {lam:8.0f} {mc.albedo[i]:8.4f}"
                  f" {analytic.albedo[i]:9.4f} {rel:+8.2%}")

    print()
    print(f"written to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reference"))
    parser.add_argument("--constants", type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="also spot-check a few wavelengths with the transport engine",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="evaluate every Mie point from scratch instead of reusing the table",
    )
    args = parser.parse_args()

    # Nine curves over one wavelength grid, and the grain populations repeat
    # between research questions 1 and 2. Cached, the second and later runs of
    # this script do no Mie work at all.
    with ExitStack() as stack:
        solver = MiepythonSolver()
        if not args.no_cache:
            solver = stack.enter_context(
                CachedMieSolver(
                    solver, frozen_paths=frozen_table_paths(solver, MIE_TABLE_DIR)
                )
            )
        status = generate(args, solver)
        if isinstance(solver, CachedMieSolver):
            print(f"mie cache         : {solver.hits} reused,"
                  f" {solver.misses} evaluated ({solver.cache_path})")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
